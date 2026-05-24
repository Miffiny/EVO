#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include <limits>
#include <omp.h>

#include "config.hpp"

struct ProblemData {
    int32_t n;
    int32_t k;
    float scale;
    std::vector<float> points;
    std::vector<int32_t> neighbors;
    std::vector<float> distances;
    std::vector<float> heuristic;
};

bool load_problem_bin(const std::string& filepath, ProblemData& data) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filepath << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&data.n), sizeof(data.n));
    file.read(reinterpret_cast<char*>(&data.k), sizeof(data.k));
    file.read(reinterpret_cast<char*>(&data.scale), sizeof(data.scale));

    int n = data.n;
    int k = data.k;

    data.points.resize(n * 2);
    data.neighbors.resize(n * k);
    data.distances.resize(n * k);
    data.heuristic.resize(n * k);

    file.read(reinterpret_cast<char*>(data.points.data()), data.points.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(data.neighbors.data()), data.neighbors.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(data.distances.data()), data.distances.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(data.heuristic.data()), data.heuristic.size() * sizeof(float));

    return true;
}

inline float calc_dist(const ProblemData& data, int u, int v) {
    float dx = data.points[u * 2] - data.points[v * 2];
    float dy = data.points[u * 2 + 1] - data.points[v * 2 + 1];
    return std::sqrt(dx * dx + dy * dy);
}

double route_length(const ProblemData& data, const std::vector<int32_t>& route, bool closed) {
    if (route.size() < 2) return 0.0;
    double total_dist = 0.0;
    for (size_t i = 0; i < route.size() - 1; ++i) {
        total_dist += calc_dist(data, route[i], route[i + 1]);
    }
    if (closed) {
        total_dist += calc_dist(data, route.back(), route.front());
    }
    return total_dist;
}

std::vector<int32_t> build_local_neighbor_indices(const std::vector<int32_t>& neighbors, int n, int k) {
    std::vector<int32_t> indices(n * n, -1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            int32_t global_neighbor = neighbors[i * k + j];
            indices[i * n + global_neighbor] = j;
        }
    }
    return indices;
}

bool apply_node_exchange(std::vector<int32_t>& route, const ProblemData& data, bool closed) {
    bool improved_global = false;
    int k = route.size();

    std::vector<bool> in_route(data.n, false);
    for (int v : route) in_route[v] = true;

    for (int i = 0; i < k; ++i) {
        int prev = (i == 0) ? (closed ? route.back() : -1) : route[i - 1];
        int next = (i == k - 1) ? (closed ? route.front() : -1) : route[i + 1];
        int curr = route[i];

        double d_old = 0.0;
        if (prev != -1) d_old += calc_dist(data, prev, curr);
        if (next != -1) d_old += calc_dist(data, curr, next);

        int best_u = -1;
        double best_saving = 0.0;

        for (int u = 0; u < data.n; ++u) {
            if (in_route[u]) continue;
            double d_new = 0.0;
            if (prev != -1) d_new += calc_dist(data, prev, u);
            if (next != -1) d_new += calc_dist(data, u, next);

            if (d_old - d_new > best_saving + 1e-4) {
                best_saving = d_old - d_new;
                best_u = u;
            }
        }

        if (best_u != -1) {
            in_route[curr] = false;
            in_route[best_u] = true;
            route[i] = best_u;
            improved_global = true;
        }
    }
    return improved_global;
}

bool apply_2opt(std::vector<int32_t>& route, const ProblemData& data, bool closed) {
    bool improved = true;
    bool global_improved = false;
    int n = route.size();

    while (improved) {
        improved = false;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 2; j < n; ++j) {
                if (i == 0 && j == n - 1 && !closed) continue;

                int u1 = route[i];
                int v1 = route[i + 1];
                int u2 = route[j];
                int v2 = (j + 1 < n) ? route[j + 1] : (closed ? route[0] : -1);

                double d_old = calc_dist(data, u1, v1) + (v2 != -1 ? calc_dist(data, u2, v2) : 0);
                double d_new = calc_dist(data, u1, u2) + (v2 != -1 ? calc_dist(data, v1, v2) : 0);

                if (d_new < d_old - 1e-4) {
                    std::reverse(route.begin() + i + 1, route.begin() + j + 1);
                    improved = true;
                    global_improved = true;
                }
            }
        }
    }
    return global_improved;
}

bool apply_or_opt(std::vector<int32_t>& route, const ProblemData& data, bool closed) {
    bool improved = true;
    bool global_improved = false;
    int n = route.size();

    while (improved) {
        improved = false;
        for (int i = 0; i < n; ++i) {
            int curr = route[i];
            int prev = (i == 0) ? (closed ? route.back() : -1) : route[i - 1];
            int next = (i == n - 1) ? (closed ? route.front() : -1) : route[i + 1];

            double d_removed = 0;
            if (prev != -1) d_removed += calc_dist(data, prev, curr);
            if (next != -1) d_removed += calc_dist(data, curr, next);
            if (prev != -1 && next != -1) d_removed -= calc_dist(data, prev, next);

            int best_pos = -1;
            double best_saving = 0;

            for (int j = 0; j < n; ++j) {
                if (j == i || j == i - 1 || (i == 0 && j == n - 1)) continue;

                int target_prev = route[j];
                int target_next = (j + 1 < n) ? route[j + 1] : (closed ? route[0] : -1);

                double d_added = calc_dist(data, target_prev, curr);
                if (target_next != -1) d_added += calc_dist(data, curr, target_next);
                if (target_next != -1) d_added -= calc_dist(data, target_prev, target_next);

                if (d_removed - d_added > best_saving + 1e-4) {
                    best_saving = d_removed - d_added;
                    best_pos = j;
                }
            }

            if (best_pos != -1) {
                route.erase(route.begin() + i);
                if (best_pos > i) best_pos--;
                route.insert(route.begin() + best_pos + 1, curr);
                improved = true;
                global_improved = true;
                break;
            }
        }
    }
    return global_improved;
}

int32_t nearest_unvisited_on_demand(const ProblemData& data, int32_t current, const std::vector<bool>& visited) {
    float c_x = data.points[current * 2];
    float c_y = data.points[current * 2 + 1];

    float min_dist_sq = std::numeric_limits<float>::infinity();
    int32_t nearest = -1;

    for (int32_t i = 0; i < data.n; ++i) {
        if (visited[i] || i == current) continue;

        float dx = data.points[i * 2] - c_x;
        float dy = data.points[i * 2 + 1] - c_y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest = i;
        }
    }
    return nearest;
}

int32_t choose_next_node(int32_t current, const std::vector<bool>& visited, const ProblemData& data,
                         const std::vector<float>& heuristic_weights, const std::vector<float>& tau,
                         std::mt19937& gen, const cfg::MMASParameters& params) {
    int k = data.k;

    std::vector<int32_t> candidates;
    std::vector<double> weights;
    candidates.reserve(k);
    weights.reserve(k);

    for (int j = 0; j < k; ++j) {
        int32_t node = data.neighbors[current * k + j];
        if (!visited[node]) {
            candidates.push_back(node);

            float pheromone = tau[current * k + j];
            float heuristic = heuristic_weights[current * k + j];
            double weight = (params.alpha == 1.0)
                ? pheromone * heuristic
                : std::pow(pheromone, params.alpha) * heuristic;

            weights.push_back(weight);
        }
    }

    if (candidates.empty()) return -1;

    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int sample_idx = dist(gen);

    if (weights[sample_idx] <= 0.0 || std::isnan(weights[sample_idx])) {
        std::uniform_int_distribution<int> unif(0, static_cast<int>(candidates.size()) - 1);
        return candidates[unif(gen)];
    }

    return candidates[sample_idx];
}

std::vector<int32_t> construct_route(const ProblemData& data, const std::vector<float>& heuristic_weights,
                                     const std::vector<float>& tau, int selected_count, std::mt19937& gen,
                                     const cfg::MMASParameters& params) {
    std::vector<bool> visited(data.n, false);
    std::vector<int32_t> route;
    route.reserve(selected_count);

    std::uniform_int_distribution<int32_t> start_dist(0, data.n - 1);
    int32_t start = start_dist(gen);

    route.push_back(start);
    visited[start] = true;
    int32_t current = start;

    while (route.size() < static_cast<size_t>(selected_count)) {
        int32_t next_node = choose_next_node(current, visited, data, heuristic_weights, tau, gen, params);
        if (next_node == -1) {
            next_node = nearest_unvisited_on_demand(data, current, visited);
        }

        if (next_node == -1) break;

        route.push_back(next_node);
        visited[next_node] = true;
        current = next_node;
    }

    return route;
}

using AntResult = std::pair<double, std::vector<int32_t>>;

void update_best_buffer(std::vector<AntResult>& buffer, AntResult result, size_t limit) {
    if (limit == 0) return;

    auto better = [](const AntResult& a, const AntResult& b) {
        return a.first < b.first;
    };

    if (buffer.size() < limit) {
        buffer.push_back(std::move(result));
        std::sort(buffer.begin(), buffer.end(), better);
        return;
    }

    if (result.first >= buffer.back().first) {
        return;
    }

    buffer.back() = std::move(result);
    std::sort(buffer.begin(), buffer.end(), better);
}

std::pair<std::vector<int32_t>, double> run_mmas(const ProblemData& data, const cfg::MMASParameters& params, bool verbose = true) {
    int n = data.n;
    int k = data.k;
    int selected_count = static_cast<int>(std::ceil(params.coverage * n));

    std::vector<float> heuristic_weights(n * k);
    for (int i = 0; i < n * k; ++i) {
        heuristic_weights[i] = std::pow(data.heuristic[i], params.beta);
    }

    auto neighbor_indices = build_local_neighbor_indices(data.neighbors, n, k);
    std::vector<float> tau(n * k, params.initial_tau);

    std::vector<int32_t> global_best_route;
    double global_best_length = std::numeric_limits<double>::infinity();

    std::vector<std::mt19937> thread_gens;
    for (int i = 0; i < omp_get_max_threads(); ++i) {
        thread_gens.emplace_back(params.seed + static_cast<uint32_t>(i * 1337));
    }

    for (int iteration = 0; iteration < params.iterations; ++iteration) {
        size_t best_buffer_limit = static_cast<size_t>(std::min(std::max(params.ls_top_ants, 1), params.ants));
        std::vector<AntResult> best_buffer;
        best_buffer.reserve(best_buffer_limit);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::mt19937& gen = thread_gens[thread_id];

            #pragma omp for
            for (int ant = 0; ant < params.ants; ++ant) {
                auto route = construct_route(data, heuristic_weights, tau, selected_count, gen, params);
                double length = route_length(data, route, params.closed_tour);

                #pragma omp critical(best_buffer_update)
                {
                    update_best_buffer(best_buffer, {length, std::move(route)}, best_buffer_limit);
                }
            }
        }

        double iteration_best_length = std::numeric_limits<double>::infinity();
        std::vector<int32_t> iteration_best_route;

        for (auto& [len, route] : best_buffer) {

            if (params.use_node_exchange) apply_node_exchange(route, data, params.closed_tour);
            if (params.use_2opt)          apply_2opt(route, data, params.closed_tour);
            if (params.use_or_opt)        apply_or_opt(route, data, params.closed_tour);

            len = route_length(data, route, params.closed_tour);

            if (len < iteration_best_length) {
                iteration_best_length = len;
                iteration_best_route = route;
            }
        }

        if (iteration_best_length < global_best_length) {
            global_best_length = iteration_best_length;
            global_best_route = iteration_best_route;
        }

        double current_tau_max = params.tau_max;
        double current_tau_min = params.tau_min;

        if (global_best_length < std::numeric_limits<double>::infinity()) {
            double Q = 10000.0;
            current_tau_max = Q / (params.evaporation * global_best_length);
            current_tau_min = current_tau_max / params.tau_gap;
        }

        for (int i = 0; i < n * k; ++i) {
            tau[i] *= (1.0 - params.evaporation);
            if (tau[i] < current_tau_min) tau[i] = current_tau_min;
        }

        const std::vector<int32_t>* deposit_route = &iteration_best_route;
        double deposit_length = iteration_best_length;

        if (params.update_strategy == cfg::PheromoneUpdateStrategy::BestSoFar) {
            deposit_route = &global_best_route;
            deposit_length = global_best_length;
        }

        if (deposit_length < std::numeric_limits<double>::infinity()) {
            double delta = 10000.0 / deposit_length;

            auto add_pheromone = [&](int32_t u, int32_t v) {
                int32_t loc_u_to_v = neighbor_indices[u * n + v];
                if (loc_u_to_v != -1) {
                    tau[u * k + loc_u_to_v] = std::min(tau[u * k + loc_u_to_v] + delta, current_tau_max);
                }
                int32_t loc_v_to_u = neighbor_indices[v * n + u];
                if (loc_v_to_u != -1) {
                    tau[v * k + loc_v_to_u] = std::min(tau[v * k + loc_v_to_u] + delta, current_tau_max);
                }
            };

            for (size_t i = 0; i < deposit_route->size() - 1; ++i) {
                add_pheromone((*deposit_route)[i], (*deposit_route)[i + 1]);
            }
            if (params.closed_tour && deposit_route->size() > 1) {
                add_pheromone(deposit_route->back(), deposit_route->front());
            }
        }

        if (verbose && (iteration + 1) % 10 == 0) {
            std::cout << "Iteration=" << (iteration + 1)
                      << " | Iter_Best=" << iteration_best_length
                      << " | Global_Best=" << global_best_length << std::endl;
        }
    }

    return {global_best_route, global_best_length};
}

std::vector<AntResult> run_mmas_collect_top_routes(const ProblemData& data, const cfg::MMASParameters& params, int buffer_size, bool verbose = true) {
    int n = data.n;
    int k = data.k;
    int selected_count = static_cast<int>(std::ceil(params.coverage * n));
    size_t collect_limit = static_cast<size_t>(std::max(buffer_size, 1));

    std::vector<float> heuristic_weights(n * k);
    for (int i = 0; i < n * k; ++i) {
        heuristic_weights[i] = std::pow(data.heuristic[i], params.beta);
    }

    auto neighbor_indices = build_local_neighbor_indices(data.neighbors, n, k);
    std::vector<float> tau(n * k, params.initial_tau);

    std::vector<int32_t> global_best_route;
    double global_best_length = std::numeric_limits<double>::infinity();
    std::vector<AntResult> collected_routes;
    collected_routes.reserve(collect_limit);

    std::vector<std::mt19937> thread_gens;
    for (int i = 0; i < omp_get_max_threads(); ++i) {
        thread_gens.emplace_back(params.seed + static_cast<uint32_t>(i * 1337));
    }

    for (int iteration = 0; iteration < params.iterations; ++iteration) {
        size_t iteration_limit = static_cast<size_t>(std::min(std::max(params.ls_top_ants, 1), params.ants));
        std::vector<AntResult> iteration_buffer;
        iteration_buffer.reserve(iteration_limit);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::mt19937& gen = thread_gens[thread_id];

            #pragma omp for
            for (int ant = 0; ant < params.ants; ++ant) {
                auto route = construct_route(data, heuristic_weights, tau, selected_count, gen, params);
                double length = route_length(data, route, params.closed_tour);
                auto collected_route = route;

                #pragma omp critical(best_buffer_update)
                {
                    update_best_buffer(iteration_buffer, {length, std::move(route)}, iteration_limit);
                    update_best_buffer(collected_routes, {length, std::move(collected_route)}, collect_limit);
                }
            }
        }

        if (iteration_buffer.empty()) continue;

        double iteration_best_length = iteration_buffer.front().first;
        const std::vector<int32_t>& iteration_best_route = iteration_buffer.front().second;

        if (iteration_best_length < global_best_length) {
            global_best_length = iteration_best_length;
            global_best_route = iteration_best_route;
        }

        double current_tau_max = params.tau_max;
        double current_tau_min = params.tau_min;

        if (global_best_length < std::numeric_limits<double>::infinity()) {
            double Q = 10000.0;
            current_tau_max = Q / (params.evaporation * global_best_length);
            current_tau_min = current_tau_max / params.tau_gap;
        }

        for (int i = 0; i < n * k; ++i) {
            tau[i] *= (1.0 - params.evaporation);
            if (tau[i] < current_tau_min) tau[i] = current_tau_min;
        }

        const std::vector<int32_t>* deposit_route = &iteration_best_route;
        double deposit_length = iteration_best_length;

        if (params.update_strategy == cfg::PheromoneUpdateStrategy::BestSoFar) {
            deposit_route = &global_best_route;
            deposit_length = global_best_length;
        }

        if (deposit_length < std::numeric_limits<double>::infinity()) {
            double delta = 10000.0 / deposit_length;

            auto add_pheromone = [&](int32_t u, int32_t v) {
                int32_t loc_u_to_v = neighbor_indices[u * n + v];
                if (loc_u_to_v != -1) {
                    tau[u * k + loc_u_to_v] = std::min(tau[u * k + loc_u_to_v] + delta, current_tau_max);
                }
                int32_t loc_v_to_u = neighbor_indices[v * n + u];
                if (loc_v_to_u != -1) {
                    tau[v * k + loc_v_to_u] = std::min(tau[v * k + loc_v_to_u] + delta, current_tau_max);
                }
            };

            for (size_t i = 0; i < deposit_route->size() - 1; ++i) {
                add_pheromone((*deposit_route)[i], (*deposit_route)[i + 1]);
            }
            if (params.closed_tour && deposit_route->size() > 1) {
                add_pheromone(deposit_route->back(), deposit_route->front());
            }
        }

        if (verbose && (iteration + 1) % 10 == 0) {
            std::cout << "Iteration=" << (iteration + 1)
                      << " | Iter_Best=" << iteration_best_length
                      << " | Global_Best=" << global_best_length << std::endl;
        }
    }

    return collected_routes;
}

double optimize_route_buffer(const ProblemData& data, const std::vector<AntResult>& route_buffer, const cfg::MMASParameters& params) {
    double best_length = std::numeric_limits<double>::infinity();

    for (const auto& route_result : route_buffer) {
        auto route = route_result.second;

        if (params.use_node_exchange) apply_node_exchange(route, data, params.closed_tour);
        if (params.use_2opt)          apply_2opt(route, data, params.closed_tour);
        if (params.use_or_opt)        apply_or_opt(route, data, params.closed_tour);

        double length = route_length(data, route, params.closed_tour);
        if (length < best_length) {
            best_length = length;
        }
    }

    return best_length;
}
