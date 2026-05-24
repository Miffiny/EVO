#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <optional>
#include <sstream>
#include <iomanip>
#include <cstdint>

namespace cfg {

    enum class PheromoneUpdateStrategy {
        BestSoFar,
        IterationBest
    };

    inline std::string to_string(PheromoneUpdateStrategy strategy) {
        switch (strategy) {
            case PheromoneUpdateStrategy::BestSoFar:
                return "best_so_far";
            case PheromoneUpdateStrategy::IterationBest:
                return "iteration_best";
        }
        return "iteration_best";
    }

    inline uint32_t random_seed() {
        std::random_device rd;
        std::mt19937 gen(rd());
        return gen();
    }

    struct MMASParameters {
        double coverage = 0.50;
        int k = 60;
        int ants = 64;
        int iterations = 200;
        double alpha = 2.0;
        double beta = 6.0;
        double evaporation = 0.2;
        double tau_min = 0.01;
        double tau_max = 10.0;
        double initial_tau = 1.0;
        uint32_t seed = random_seed();
        bool closed_tour = true;
        double tau_gap = 50.0;
        bool use_2opt = false;
        bool use_or_opt = false;
        bool use_node_exchange = false;
        int ls_top_ants = 5;
        PheromoneUpdateStrategy update_strategy = PheromoneUpdateStrategy::IterationBest;
    };

    const MMASParameters DEFAULT_PARAMETERS;

    inline std::string format_val(double val) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << val;
        std::string str = out.str();
        str.erase(str.find_last_not_of('0') + 1, std::string::npos);
        if (str.back() == '.') str.pop_back();
        return str;
    }

    inline std::vector<double> parameter_range(double start, double stop, double step) {
        std::vector<double> values;
        for (double current = start; current <= stop + 1e-6; current += step) {
            values.push_back(current);
        }
        return values;
    }

    struct ExperimentRanges {
        std::optional<std::vector<int>> ants;
        std::optional<std::vector<double>> alpha;
        std::optional<std::vector<double>> beta;
        std::optional<std::vector<double>> evaporation;
        std::optional<std::vector<double>> tau_gap;
        std::optional<std::vector<double>> coverage;
        std::optional<std::vector<PheromoneUpdateStrategy>> update_strategy;
    };

    using ConfigRun = std::pair<std::string, MMASParameters>;

    inline std::vector<ConfigRun> make_ofat_combinations(const ExperimentRanges& ranges) {
        std::vector<ConfigRun> combinations;

        combinations.push_back({"baseline", DEFAULT_PARAMETERS});

        if (ranges.alpha) {
            for (double val : *ranges.alpha) {
                if (val == DEFAULT_PARAMETERS.alpha) continue;
                MMASParameters p = DEFAULT_PARAMETERS;
                p.alpha = val;
                combinations.push_back({"alpha=" + format_val(val), p});
            }
        }

        if (ranges.beta) {
            for (double val : *ranges.beta) {
                if (val == DEFAULT_PARAMETERS.beta) continue;
                MMASParameters p = DEFAULT_PARAMETERS;
                p.beta = val;
                combinations.push_back({"beta=" + format_val(val), p});
            }
        }

        if (ranges.evaporation) {
            for (double val : *ranges.evaporation) {
                if (val == DEFAULT_PARAMETERS.evaporation) continue;
                MMASParameters p = DEFAULT_PARAMETERS;
                p.evaporation = val;
                combinations.push_back({"evaporation=" + format_val(val), p});
            }
        }

        if (ranges.ants) {
            for (int val : *ranges.ants) {
                if (val == DEFAULT_PARAMETERS.ants) continue;
                MMASParameters p = DEFAULT_PARAMETERS;
                p.ants = val;
                combinations.push_back({"ants=" + std::to_string(val), p});
            }
        }

        if (ranges.update_strategy) {
            for (PheromoneUpdateStrategy val : *ranges.update_strategy) {
                if (val == DEFAULT_PARAMETERS.update_strategy) continue;
                MMASParameters p = DEFAULT_PARAMETERS;
                p.update_strategy = val;
                combinations.push_back({"update_strategy=" + to_string(val), p});
            }
        }

        return combinations;
    }

    inline std::vector<ConfigRun> make_grid_combinations(
        const std::vector<double>& alphas,
        const std::vector<double>& betas,
        const std::vector<double>& evaporations)
    {
        std::vector<ConfigRun> combinations;

        for (double a : alphas) {
            for (double b : betas) {
                for (double e : evaporations) {
                    MMASParameters p = DEFAULT_PARAMETERS;
                    p.alpha = a;
                    p.beta = b;
                    p.evaporation = e;

                    std::string name = "alpha=" + format_val(a) +
                                     "_beta=" + format_val(b) +
                                     "_evaporation=" + format_val(e);

                    combinations.push_back({name, p});
                }
            }
        }
        return combinations;
    }
}
