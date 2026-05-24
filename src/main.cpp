#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

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

using RouteBuffer = std::vector<std::pair<double, std::vector<int32_t>>>;

bool load_problem_bin(const std::string& filepath, ProblemData& data);
std::pair<std::vector<int32_t>, double> run_mmas(
    const ProblemData& data,
    const cfg::MMASParameters& params,
    bool verbose = false
);
RouteBuffer run_mmas_collect_top_routes(
    const ProblemData& data,
    const cfg::MMASParameters& params,
    int buffer_size,
    bool verbose = false
);
double optimize_route_buffer(
    const ProblemData& data,
    const RouteBuffer& route_buffer,
    const cfg::MMASParameters& params
);

namespace fs = std::filesystem;

enum class RunMode {
    Single,
    Ofat,
    Pair,
    Opt,
    All
};

struct PairParameter {
    std::string name;
    std::vector<double> values;
};

fs::path find_project_root_from(fs::path current) {
    while (true) {
        if (fs::exists(current / "src") && fs::exists(current / "data")) {
            return current;
        }

        fs::path parent = current.parent_path();
        if (parent == current || parent.empty()) {
            return fs::current_path();
        }
        current = parent;
    }
}

fs::path find_project_root(const char* executable_path) {
    std::vector<fs::path> candidates = {fs::current_path()};

    if (executable_path != nullptr && std::string(executable_path).size() > 0) {
        candidates.push_back(fs::absolute(executable_path).parent_path());
    }

    for (const fs::path& candidate : candidates) {
        fs::path root = find_project_root_from(candidate);
        if (fs::exists(root / "src") && fs::exists(root / "data")) {
            return root;
        }
    }

    return fs::current_path();
}

std::vector<fs::path> find_precomputed_files(const fs::path& precomputed_dir) {
    std::vector<fs::path> files;

    if (!fs::exists(precomputed_dir)) {
        return files;
    }

    for (const auto& entry : fs::directory_iterator(precomputed_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

void write_result_row(
    std::ofstream& csv_file,
    const fs::path& bin_path,
    const std::string& combo_name,
    int trial,
    const cfg::MMASParameters& params,
    double best_length,
    double elapsed_seconds
) {
    std::ostringstream row;
    row << bin_path.filename().string() << ","
        << combo_name << ","
        << trial << ","
        << cfg::to_string(params.update_strategy) << ","
        << cfg::format_val(params.coverage) << ","
        << cfg::format_val(params.alpha) << ","
        << cfg::format_val(params.beta) << ","
        << cfg::format_val(params.evaporation) << ","
        << params.ants << ","
        << cfg::format_val(params.tau_gap) << ","
        << params.use_2opt << ","
        << params.use_or_opt << ","
        << params.use_node_exchange << ","
        << params.ls_top_ants << ","
        << std::fixed << std::setprecision(4) << best_length << ","
        << std::setprecision(6) << elapsed_seconds << "\n";

    csv_file << row.str();
}

bool run_experiments_for_file(
    const fs::path& bin_path,
    const fs::path& csv_output_path,
    const std::vector<cfg::ConfigRun>& experiments,
    int num_trials
) {
    ProblemData data;
    std::cout << "[dataset] Loading " << bin_path.string() << std::endl;

    if (!load_problem_bin(bin_path.string(), data)) {
        std::cerr << "[error] Failed to load graph data from " << bin_path.string() << std::endl;
        return false;
    }

    std::ofstream csv_file(csv_output_path);
    if (!csv_file.is_open()) {
        std::cerr << "[error] Failed to create results file " << csv_output_path.string() << std::endl;
        return false;
    }

    csv_file << "dataset,combo_name,trial_id,update_strategy,coverage,alpha,beta,evaporation,"
             << "ants,tau_gap,use_2opt,use_or_opt,use_node_exchange,ls_top_ants,"
             << "best_length,elapsed_seconds\n";

    std::random_device rd;
    std::mt19937_64 seed_generator(rd());

    const int total_tasks = static_cast<int>(experiments.size()) * num_trials;
    int completed_tasks = 0;

    std::cout << "[dataset] N=" << data.n
              << " | configs=" << experiments.size()
              << " | runs=" << total_tasks
              << " | output=" << csv_output_path.string() << std::endl;

    for (const auto& [combo_name, base_params] : experiments) {
        std::cout << "  >> " << combo_name << std::endl;

        for (int trial = 1; trial <= num_trials; ++trial) {
            cfg::MMASParameters trial_params = base_params;
            trial_params.seed = static_cast<uint32_t>(seed_generator());

            const auto start_time = std::chrono::high_resolution_clock::now();
            const auto result = run_mmas(data, trial_params, false);
            const auto end_time = std::chrono::high_resolution_clock::now();

            const std::chrono::duration<double> elapsed = end_time - start_time;
            const double best_length = result.second;

            write_result_row(csv_file, bin_path, combo_name, trial, trial_params, best_length, elapsed.count());

            csv_file.flush();

            ++completed_tasks;
            std::cout << "     [" << completed_tasks << "/" << total_tasks << "] "
                      << "trial=" << trial
                      << " | best=" << std::setprecision(4) << best_length
                      << " | time=" << std::setprecision(2) << elapsed.count() << "s"
                      << std::endl;
        }
    }

    std::cout << "[done] Saved " << csv_output_path.string() << "\n" << std::endl;
    return true;
}

std::vector<cfg::ConfigRun> make_ofat() {
    std::vector<cfg::ConfigRun> experiments;

    std::vector<double> coverages = {0.20, 0.50, 0.80};
    std::vector<cfg::PheromoneUpdateStrategy> strategies = {
        cfg::PheromoneUpdateStrategy::IterationBest,
        cfg::PheromoneUpdateStrategy::BestSoFar
    };

    std::vector<double> alphas   = {0.2, 0.5, 1.0, 2.0, 3.0, 5.0};
    std::vector<double> betas    = {2.0, 3.0, 4.0, 5.0, 7.0};
    std::vector<double> evaps    = {0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8};
    std::vector<int>    ants     = {16, 32, 64, 100};
    std::vector<double> tau_gaps = {10.0, 50.0, 100.0, 250.0};

    for (double cov : coverages) {
        for (auto strat : strategies) {

            cfg::MMASParameters base = cfg::DEFAULT_PARAMETERS;
            base.coverage = cov;
            base.update_strategy = strat;

            std::string prefix = cfg::to_string(strat) + "_cov" + std::to_string(static_cast<int>(cov * 100)) + "_";

            experiments.push_back({prefix + "baseline", base});

            for (double a : alphas) {
                if (a == base.alpha) continue;
                auto p = base; p.alpha = a;
                experiments.push_back({prefix + "alpha=" + cfg::format_val(a), p});
            }

            for (double b : betas) {
                if (b == base.beta) continue;
                auto p = base; p.beta = b;
                experiments.push_back({prefix + "beta=" + cfg::format_val(b), p});
            }

            for (double e : evaps) {
                if (e == base.evaporation) continue;
                auto p = base; p.evaporation = e;
                experiments.push_back({prefix + "evaporation=" + cfg::format_val(e), p});
            }

            for (int a : ants) {
                if (a == base.ants) continue;
                auto p = base; p.ants = a;
                experiments.push_back({prefix + "ants=" + std::to_string(a), p});
            }

            for (double tg : tau_gaps) {
                if (tg == base.tau_gap) continue;
                auto p = base; p.tau_gap = tg;
                experiments.push_back({prefix + "tau_gap=" + cfg::format_val(tg), p});
            }
        }
    }
    return experiments;
}

void set_pair_parameter(cfg::MMASParameters& params, const std::string& name, double value) {
    if (name == "alpha") {
        params.alpha = value;
        return;
    }
    if (name == "beta") {
        params.beta = value;
        return;
    }
    if (name == "evaporation") {
        params.evaporation = value;
        return;
    }
    if (name == "tau_gap") {
        params.tau_gap = value;
        return;
    }
    if (name == "coverage") {
        params.coverage = value;
        return;
    }
    if (name == "ants") {
        params.ants = static_cast<int>(std::lround(value));
        return;
    }
    if (name == "iterations") {
        params.iterations = static_cast<int>(std::lround(value));
        return;
    }

    throw std::invalid_argument("Unsupported pair parameter: " + name);
}

PairParameter make_pair_parameter(const std::string& name) {
    if (name == "alpha") return {name, {1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4}};
    if (name == "beta") return {name, {5.0, 6.0, 7.0, 8.0}};
    if (name == "evaporation") return {name, {0.05, 0.1, 0.2, 0.4, 0.6}};
    if (name == "tau_gap") return {name, {50.0, 100.0, 250.0, 500.0}};
    if (name == "coverage") return {name, {0.30, 0.50, 0.70, 0.90}};
    if (name == "ants") return {name, {32.0, 64.0, 96.0, 128.0}};
    if (name == "iterations") return {name, {100.0, 200.0, 300.0, 400.0}};

    throw std::invalid_argument("Unsupported pair parameter: " + name);
}

std::string pair_output_prefix(const PairParameter& first, const PairParameter& second) {
    return "pairs_" + first.name + "_" + second.name + "_";
}

std::vector<cfg::ConfigRun> make_pair_experiments(
    const PairParameter& first = make_pair_parameter("alpha"),
    const PairParameter& second = make_pair_parameter("beta")
) {
    std::vector<cfg::ConfigRun> experiments;

    cfg::MMASParameters base = cfg::DEFAULT_PARAMETERS;
    base.coverage = 0.50;
    base.ants = 64;
    base.iterations = 300;
    base.evaporation = 0.2;
    base.tau_gap = 250.0;
    base.closed_tour = true;
    base.update_strategy = cfg::PheromoneUpdateStrategy::IterationBest;

    for (double first_value : first.values) {
        for (double second_value : second.values) {
            cfg::MMASParameters p = base;
            set_pair_parameter(p, first.name, first_value);
            set_pair_parameter(p, second.name, second_value);

            std::string name = first.name + "=" + cfg::format_val(first_value) +
                             "_" + second.name + "=" + cfg::format_val(second_value);

            experiments.push_back({name, p});
        }
    }

    return experiments;
}

cfg::MMASParameters make_optimization_base_parameters() {
    cfg::MMASParameters base = cfg::DEFAULT_PARAMETERS;
    base.coverage = 0.50;
    base.ants = 64;
    base.iterations = 300;
    base.alpha = 2.0;
    base.beta = 5.0;
    base.evaporation = 0.2;
    base.tau_gap = 250.0;
    base.closed_tour = true;
    base.update_strategy = cfg::PheromoneUpdateStrategy::IterationBest;
    base.use_2opt = false;
    base.use_or_opt = false;
    base.use_node_exchange = false;
    base.ls_top_ants = 7;

    return base;
}

std::vector<cfg::ConfigRun> make_optimization_experiments() {
    std::vector<cfg::ConfigRun> experiments;
    cfg::MMASParameters base = make_optimization_base_parameters();

    cfg::MMASParameters two_opt = base;
    two_opt.use_2opt = true;
    experiments.push_back({"use_2opt=true", two_opt});

    cfg::MMASParameters or_opt = base;
    or_opt.use_or_opt = true;
    experiments.push_back({"use_or_opt=true", or_opt});

    cfg::MMASParameters node_exchange = base;
    node_exchange.use_node_exchange = true;
    experiments.push_back({"use_node_exchange=true", node_exchange});

    cfg::MMASParameters node_exchange_2opt = base;
    node_exchange_2opt.use_node_exchange = true;
    node_exchange_2opt.use_2opt = true;
    experiments.push_back({"use_node_exchange=true_use_2opt=true", node_exchange_2opt});

    return experiments;
}

int run_config_set_for_all_files(
    const std::vector<fs::path>& bin_files,
    const fs::path& results_dir,
    const std::vector<cfg::ConfigRun>& experiments,
    int num_trials,
    const std::string& output_prefix,
    const std::string& mode_name
) {
    std::cout << "[info] Found datasets: " << bin_files.size() << std::endl;
    std::cout << "[info] Mode: " << mode_name << std::endl;
    std::cout << "[info] Configurations per dataset: " << experiments.size() << std::endl;
    std::cout << "[info] Trials per configuration: " << num_trials << "\n" << std::endl;

    int failed_files = 0;

    for (const fs::path& bin_path : bin_files) {
        const fs::path csv_output_path =
            results_dir / (output_prefix + bin_path.stem().string() + ".csv");

        if (!run_experiments_for_file(bin_path, csv_output_path, experiments, num_trials)) {
            ++failed_files;
        }
    }

    if (failed_files > 0) {
        std::cerr << "[finished] Completed with " << failed_files << " failed dataset(s)." << std::endl;
        return 1;
    }

    std::cout << "[success] " << mode_name << " completed for all datasets." << std::endl;
    return 0;
}

bool run_optimization_postprocessing_for_file(
    const fs::path& bin_path,
    const fs::path& csv_output_path,
    const std::vector<cfg::ConfigRun>& optimizations,
    int num_trials,
    int buffer_size
) {
    ProblemData data;
    std::cout << "[dataset] Loading " << bin_path.string() << std::endl;

    if (!load_problem_bin(bin_path.string(), data)) {
        std::cerr << "[error] Failed to load graph data from " << bin_path.string() << std::endl;
        return false;
    }

    std::ofstream csv_file(csv_output_path);
    if (!csv_file.is_open()) {
        std::cerr << "[error] Failed to create results file " << csv_output_path.string() << std::endl;
        return false;
    }

    csv_file << "dataset,combo_name,trial_id,update_strategy,coverage,alpha,beta,evaporation,"
             << "ants,tau_gap,use_2opt,use_or_opt,use_node_exchange,ls_top_ants,"
             << "best_length,elapsed_seconds\n";

    std::random_device rd;
    std::mt19937_64 seed_generator(rd());
    cfg::MMASParameters baseline_params = make_optimization_base_parameters();
    baseline_params.ls_top_ants = buffer_size;

    const int total_tasks = num_trials * (static_cast<int>(optimizations.size()) + 1);
    int completed_tasks = 0;

    std::cout << "[dataset] N=" << data.n
              << " | configs=" << (optimizations.size() + 1)
              << " | runs=" << total_tasks
              << " | output=" << csv_output_path.string() << std::endl;

    for (int trial = 1; trial <= num_trials; ++trial) {
        baseline_params.seed = static_cast<uint32_t>(seed_generator());

        const auto baseline_start = std::chrono::high_resolution_clock::now();
        RouteBuffer route_buffer = run_mmas_collect_top_routes(data, baseline_params, buffer_size, false);
        const auto baseline_end = std::chrono::high_resolution_clock::now();

        if (route_buffer.empty()) {
            std::cerr << "[error] Empty route buffer for " << bin_path.string()
                      << " trial=" << trial << std::endl;
            return false;
        }

        const std::chrono::duration<double> baseline_elapsed = baseline_end - baseline_start;
        const double baseline_seconds = baseline_elapsed.count();
        write_result_row(
            csv_file,
            bin_path,
            "baseline",
            trial,
            baseline_params,
            route_buffer.front().first,
            baseline_seconds
        );
        csv_file.flush();
        ++completed_tasks;
        std::cout << "     [" << completed_tasks << "/" << total_tasks << "] "
                  << "trial=" << trial
                  << " | combo=baseline"
                  << " | best=" << std::setprecision(4) << route_buffer.front().first
                  << " | total_time=" << std::setprecision(2) << baseline_seconds << "s"
                  << std::endl;

        for (const auto& [combo_name, optimization_params] : optimizations) {
            cfg::MMASParameters trial_params = optimization_params;
            trial_params.seed = baseline_params.seed;
            trial_params.ls_top_ants = buffer_size;

            const auto optimization_start = std::chrono::high_resolution_clock::now();
            const double best_length = optimize_route_buffer(data, route_buffer, trial_params);
            const auto optimization_end = std::chrono::high_resolution_clock::now();

            const std::chrono::duration<double> optimization_elapsed = optimization_end - optimization_start;
            const double postprocessing_seconds = optimization_elapsed.count();
            const double total_seconds = baseline_seconds + postprocessing_seconds;
            write_result_row(
                csv_file,
                bin_path,
                combo_name,
                trial,
                trial_params,
                best_length,
                total_seconds
            );
            csv_file.flush();

            ++completed_tasks;
            std::cout << "     [" << completed_tasks << "/" << total_tasks << "] "
                      << "trial=" << trial
                      << " | combo=" << combo_name
                      << " | best=" << std::setprecision(4) << best_length
                      << " | total_time=" << std::setprecision(2) << total_seconds << "s"
                      << " | post_time=" << postprocessing_seconds << "s"
                      << std::endl;
        }
    }

    std::cout << "[done] Saved " << csv_output_path.string() << "\n" << std::endl;
    return true;
}

bool write_single_route(
    const fs::path& output_path,
    const fs::path& bin_path,
    const std::vector<int32_t>& route,
    const ProblemData& data,
    double best_length
) {
    std::ofstream route_file(output_path);
    if (!route_file.is_open()) {
        std::cerr << "[error] Failed to create route file " << output_path.string() << std::endl;
        return false;
    }

    route_file << "dataset,best_length,route_order,node_id,x,y\n";
    route_file << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < route.size(); ++i) {
        int32_t node = route[i];
        if (node < 0 || node >= data.n) {
            std::cerr << "[error] Invalid route node " << node << " in " << bin_path.string() << std::endl;
            return false;
        }

        route_file << bin_path.filename().string() << ","
                   << best_length << ","
                   << i << ","
                   << node << ","
                   << data.points[node * 2] << ","
                   << data.points[node * 2 + 1] << "\n";
    }

    return true;
}

bool run_single_for_file(
    const fs::path& bin_path,
    const fs::path& output_path
) {
    ProblemData data;
    std::cout << "[dataset] Loading " << bin_path.string() << std::endl;

    if (!load_problem_bin(bin_path.string(), data)) {
        std::cerr << "[error] Failed to load graph data from " << bin_path.string() << std::endl;
        return false;
    }

    cfg::MMASParameters params = cfg::DEFAULT_PARAMETERS;
    params.seed = cfg::random_seed();

    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto result = run_mmas(data, params, false);
    const auto end_time = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed = end_time - start_time;
    const std::vector<int32_t>& best_route = result.first;
    const double best_length = result.second;

    if (best_route.empty()) {
        std::cerr << "[error] Empty route for " << bin_path.string() << std::endl;
        return false;
    }

    if (!write_single_route(output_path, bin_path, best_route, data, best_length)) {
        return false;
    }

    std::cout << "[single] " << bin_path.filename().string()
              << " | best=" << std::setprecision(4) << best_length
              << " | route_nodes=" << best_route.size()
              << " | time=" << std::setprecision(2) << elapsed.count() << "s"
              << " | output=" << output_path.string()
              << std::endl;

    return true;
}

int run_single_mode(
    const std::vector<fs::path>& bin_files,
    const fs::path& single_results_dir
) {
    fs::create_directories(single_results_dir);

    int failed_files = 0;
    std::cout << "[info] Found datasets: " << bin_files.size() << std::endl;
    std::cout << "[info] Mode: single" << std::endl;

    for (const fs::path& bin_path : bin_files) {
        const fs::path output_path =
            single_results_dir / ("single_route_" + bin_path.stem().string() + ".csv");

        if (!run_single_for_file(bin_path, output_path)) {
            ++failed_files;
        }
    }

    if (failed_files > 0) {
        std::cerr << "[finished] Completed with " << failed_files << " failed dataset(s)." << std::endl;
        return 1;
    }

    std::cout << "[success] single completed for all datasets." << std::endl;
    return 0;
}

int run_optimization_postprocessing_for_all_files(
    const std::vector<fs::path>& bin_files,
    const fs::path& results_dir,
    int num_trials,
    int buffer_size
) {
    std::vector<cfg::ConfigRun> optimizations = make_optimization_experiments();

    std::cout << "[info] Found datasets: " << bin_files.size() << std::endl;
    std::cout << "[info] Mode: opt" << std::endl;
    std::cout << "[info] Post-processing optimizations: " << optimizations.size() << std::endl;
    std::cout << "[info] Trials per dataset: " << num_trials << std::endl;
    std::cout << "[info] Route buffer size: " << buffer_size << "\n" << std::endl;

    int failed_files = 0;
    for (const fs::path& bin_path : bin_files) {
        const fs::path csv_output_path =
            results_dir / ("opt_results_" + bin_path.stem().string() + ".csv");

        if (!run_optimization_postprocessing_for_file(bin_path, csv_output_path, optimizations, num_trials, buffer_size)) {
            ++failed_files;
        }
    }

    if (failed_files > 0) {
        std::cerr << "[finished] Completed with " << failed_files << " failed dataset(s)." << std::endl;
        return 1;
    }

    std::cout << "[success] opt completed for all datasets." << std::endl;
    return 0;
}

int run_ofat_mode(const std::vector<fs::path>& bin_files, const fs::path& results_dir) {
    const int num_trials = 15;
    return run_config_set_for_all_files(
        bin_files,
        results_dir,
        make_ofat(),
        num_trials,
        "experiment_results_",
        "ofat"
    );
}

int run_pair_mode(
    const std::vector<fs::path>& bin_files,
    const fs::path& results_dir,
    const PairParameter& first = make_pair_parameter("alpha"),
    const PairParameter& second = make_pair_parameter("beta")
) {
    const int num_trials = 15;
    return run_config_set_for_all_files(
        bin_files,
        results_dir,
        make_pair_experiments(first, second),
        num_trials,
        pair_output_prefix(first, second),
        "pair"
    );
}

int run_opt_mode(const std::vector<fs::path>& bin_files, const fs::path& results_dir) {
    const int num_trials = 15;
    const int buffer_size = 7;
    return run_optimization_postprocessing_for_all_files(bin_files, results_dir, num_trials, buffer_size);
}

bool parse_mode(const std::string& value, RunMode& mode) {
    if (value == "single") {
        mode = RunMode::Single;
        return true;
    }
    if (value == "ofat") {
        mode = RunMode::Ofat;
        return true;
    }
    if (value == "pair") {
        mode = RunMode::Pair;
        return true;
    }
    if (value == "opt") {
        mode = RunMode::Opt;
        return true;
    }
    if (value == "all") {
        mode = RunMode::All;
        return true;
    }
    return false;
}

void print_usage(const char* executable_name) {
    std::cerr << "Usage: " << executable_name << " [single|ofat|pair|opt|all] [pair_param_1 pair_param_2]\n"
              << "  single -> single_route_<dataset>.csv in results/single_run_results\n"
              << "  ofat -> experiment_results_<dataset>.csv\n"
              << "  pair -> pairs_<param1>_<param2>_<dataset>.csv, defaults to alpha beta\n"
              << "  opt  -> opt_results_<dataset>.csv\n"
              << "  all  -> run single, ofat, pair, and opt\n"
              << "  pair parameters: alpha beta evaporation tau_gap coverage ants iterations\n";
}

int main(int argc, char* argv[]) {
    const fs::path project_root = find_project_root(argv[0]);
    const fs::path precomputed_dir = project_root / "data" / "precomputed";
    const fs::path results_dir = project_root / "results" / "run_results";
    const fs::path single_results_dir = project_root / "results" / "single_run_results";

    RunMode mode = RunMode::Ofat;
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc > 4 || (argc >= 2 && !parse_mode(argv[1], mode)) || (argc > 2 && mode != RunMode::Pair) || argc == 3) {
        print_usage(argv[0]);
        return 1;
    }

    PairParameter first_pair_parameter = make_pair_parameter("alpha");
    PairParameter second_pair_parameter = make_pair_parameter("beta");
    if (argc == 4) {
        try {
            first_pair_parameter = make_pair_parameter(argv[2]);
            second_pair_parameter = make_pair_parameter(argv[3]);
        } catch (const std::invalid_argument& error) {
            std::cerr << "[error] " << error.what() << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    const std::vector<fs::path> bin_files = find_precomputed_files(precomputed_dir);
    if (bin_files.empty()) {
        std::cerr << "[error] No .bin files found in " << precomputed_dir.string() << std::endl;
        return 1;
    }

    fs::create_directories(results_dir);

    if (mode == RunMode::Single) {
        return run_single_mode(bin_files, single_results_dir);
    }
    if (mode == RunMode::Ofat) {
        return run_ofat_mode(bin_files, results_dir);
    }
    if (mode == RunMode::Pair) {
        return run_pair_mode(bin_files, results_dir, first_pair_parameter, second_pair_parameter);
    }
    if (mode == RunMode::Opt) {
        return run_opt_mode(bin_files, results_dir);
    }

    int status = 0;
    status |= run_single_mode(bin_files, single_results_dir);
    status |= run_ofat_mode(bin_files, results_dir);
    status |= run_pair_mode(bin_files, results_dir, first_pair_parameter, second_pair_parameter);
    status |= run_opt_mode(bin_files, results_dir);
    return status == 0 ? 0 : 1;
}
