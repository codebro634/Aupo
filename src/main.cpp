#define DEBUG


#include <random>
#include "../include/Arena.h"
#include "../include/Agents/Mcts/MctsAgent.h"
#include "../include/Agents/Aupo/AupoAgent.h"
#include "../include/Agents/Mcts/MctsAgent.h"
#include "../include/Games/MDPs/SysAdmin.h"
#include "../include/Games/MDPs/GameOfLife.h"
#include "../include/Games/MDPs/Wildfire.h"
#include "../include/Games/MDPs/Tamarisk.h"
#include "../include/Games/MDPs/AcademicAdvising.h"
#include "../include/Games/MDPs/MultiArmedBandit.h"
#include "../include/Utils/Argparse.h"

void debug()
{
    const int seed = 3;
    std::mt19937 rng(static_cast<unsigned int>(seed));

    auto model = AA::Model("../resources/AcademicAdvisingCourses/2.txt",true);
    auto aupo = AUPO::AupoAgent(AUPO::AupoArgs({100, "iterations"}, {-1,2} , 1.0, -1, 4, false,0.95,  0.95, true,  false, 0, true, 0.5));
    auto results = playGames(model, 500, {&aupo}, rng, VERBOSE, {50,50}, false, true);
}

std::string extraArgs(std::map<std::string, std::string>& given_args, const std::set<std::string>& acceptable_args){
    for (auto& [key, val] : given_args) {
        if(!acceptable_args.contains(key))
            return key;
    }
    return "";
}

inline ABS::Model* getModel(const std::string& model_type, const std::vector<std::string>& m_args)
{

    std::map<std::string, std::string> model_args;
    for(auto &arg : m_args) {
        //split at '='
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            std::cout << "Invalid agent argument: " << arg << ". It must be of the form arg_name=arg_val" << std::endl;
            return nullptr;
        }
        model_args[arg.substr(0, pos)] = arg.substr(pos + 1);
    }

    ABS::Model *model = nullptr;
    std::set<std::string> acceptable_args;
    
   if (model_type == "mab") {
        assert (model_args.contains("repeats") && model_args.contains("means") && model_args.contains("stds"));
        acceptable_args = {"repeats", "means", "stds"};
        int arm_copies = std::stoi(model_args["repeats"]);
        std::vector<std::pair<double,double>> arm_distributions = {};
        std::stringstream ss(model_args["means"]);
        std::stringstream ss2(model_args["stds"]);
        double i;
        while (ss >> i){
            double j;
            ss2 >> j;
            arm_distributions.emplace_back(i, j);
            if (ss.peek() == ';')
                ss.ignore();
            if (ss2.peek() == ';')
                ss2.ignore();
        }
        model =  new MAB::Model(arm_distributions,arm_copies);
    }
    else if (model_type == "wf") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model =  new WF::Model(model_args["map"]);
    }
    else if (model_type == "tam") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model =  new TAM::Model(model_args["map"]);
    }
    else if (model_type == "sa") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model = new SA::Model(model_args["map"], false);
    }
    else if (model_type == "gol") {
        acceptable_args = {"map"};
        if(!model_args.contains("map"))
            model =  new GOL::Model();
        else
            model =  new GOL::Model(model_args["map"]);
    }
    else if (model_type == "aa") {
        assert (model_args.contains("map") && model_args.contains("dense_rewards"));
        acceptable_args = {"map", "dense_rewards"};
        model =  new AA::Model(model_args["map"], std::stoi(model_args["dense_rewards"]));
    }

    if (model != nullptr) {
        if (!extraArgs(model_args, acceptable_args).empty()) {
            std::string err_string = "Invalid model argument: " + extraArgs(model_args, acceptable_args);
            std::cout << err_string << std::endl;
            throw std::runtime_error(err_string);
        }
        return model;
    }else {
        std::cout << "Invalid model" << std::endl;
        throw std::runtime_error("Invalid model");
    }

}

Agent* getDefaultAgent()
{
    return new Mcts::MctsAgent({500, "iterations"});
}

inline Agent* getAgent(const std::string& agent_type, const std::vector<std::string>& a_args)
{

    //Parse named args
    std::map<std::string, std::string> agent_args;
    for(auto &arg   : a_args) {
        //split at '='
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            std::cout << "Invalid agent argument: " << arg << ". It must be of the form arg_name=arg_val" << std::endl;
            return nullptr;
        }
        agent_args[arg.substr(0, pos)] = arg.substr(pos + 1);
    }
    std::set<std::string> acceptable_args;

    Agent* agent;
    if(agent_type == "mcts")
    {
        assert (agent_args.contains("iterations"));
        if(agent_args.contains("wirsa"))
            assert (agent_args.contains("a") && agent_args.contains("b"));
        acceptable_args = {"iterations", "rollout_length", "discount", "num_rollouts", "dag", "dynamic_exp_factor", "expfacs"};

        int iterations = std::stoi(agent_args["iterations"]);
        int rollout_length = agent_args.find("rollout_length") == agent_args.end() ? -1 : std::stoi(agent_args["rollout_length"]);
        double discount = agent_args.find("discount") == agent_args.end() ? 1.0 : std::stod(agent_args["discount"]);
        int num_rollouts = agent_args.find("num_rollouts") == agent_args.end() ? 1: std::stoi(agent_args["num_rollouts"]);
        bool dag = agent_args.find("dag") == agent_args.end() ? false : std::stoi(agent_args["dag"]);
        bool dynamic_exp_factor = agent_args.find("dynamic_exp_factor") == agent_args.end() ? false : std::stoi(agent_args["dynamic_exp_factor"]);
        std::string exp_facs = agent_args.find("expfacs") == agent_args.end() ? "1" : agent_args["expfacs"];
        std::vector<double> expfac;
        std::stringstream ss(exp_facs);
        double i;
        while (ss >> i){
            expfac.push_back(i);
            if (ss.peek() == ';')
                ss.ignore();
        }
        auto args = Mcts::MctsArgs{.budget = {iterations, "iterations"}, .exploration_parameters = expfac, .discount = discount,
            .num_rollouts = num_rollouts,
            .rollout_length = rollout_length,
            .dag=dag,
            .dynamic_exploration_factor=dynamic_exp_factor,};
        agent =  new Mcts::MctsAgent(args);
    }

     else if (agent_type == "aupo") {

        assert (agent_args.contains("iterations"));
        acceptable_args = {"iterations", "discount", "expfacs", "rollout_length", "distribution_layers", "confidence", "confidence_std",
            "filter_by_std", "use_rollout_distribution", "dag", "min_samples", "smart_uniform_sampling", "smart_sampling_q"};

        int iterations = std::stoi(agent_args["iterations"]);
        double discount = agent_args.find("discount") == agent_args.end() ? 1.0 : std::stod(agent_args["discount"]);
        int rollout_length = agent_args.find("rollout_length") == agent_args.end() ? -1 : std::stoi(agent_args["rollout_length"]);
        std::string exp_facs = agent_args.find("expfacs") == agent_args.end() ? "2" : agent_args["expfacs"];
        std::vector<double> expfac;
        std::stringstream ss(exp_facs);
        double i;
        while (ss >> i){
            expfac.push_back(i);
            if (ss.peek() == ';')
                ss.ignore();
        }
        int distribution_layers = agent_args.find("distribution_layers") == agent_args.end() ? 3 : std::stoi(agent_args["distribution_layers"]);
        double confidence = agent_args.find("confidence") == agent_args.end() ? 0.95 : std::stod(agent_args["confidence"]);
        double confidence_std = agent_args.find("confidence_std") == agent_args.end() ? -1 : std::stod(agent_args["confidence_std"]);
        bool filter_by_std = agent_args.find("filter_by_std") == agent_args.end() ? false : std::stoi(agent_args["filter_by_std"]);
        bool use_rollout_distribution = agent_args.find("use_rollout_distribution") == agent_args.end() ? false : std::stoi(agent_args["use_rollout_distribution"]);
        bool dag = agent_args.find("dag") == agent_args.end() ? false : std::stoi(agent_args["dag"]);
        int min_samples = agent_args.find("min_samples") == agent_args.end() ? 0 : std::stoi(agent_args["min_samples"]);
        bool smart_uniform_sampling = agent_args.find("smart_uniform_sampling") == agent_args.end() ? false : std::stoi(agent_args["smart_uniform_sampling"]);
        double smart_sampling_q = agent_args.find("smart_sampling_q") == agent_args.end() ? 0.9 : std::stod(agent_args["smart_sampling_q"]);
        auto args = AUPO::AupoArgs{.budget = {iterations, "iterations"}, .exploration_parameter = expfac,
            .discount = discount, .rollout_length = rollout_length,
            .distribution_layers = distribution_layers, .use_rollout_distribution = use_rollout_distribution,
            .confidence = confidence, .confidence_std = confidence_std,
            .filter_by_std = filter_by_std,
            .dag = dag,
            .min_samples = min_samples,
            .smart_uniform_sampling = smart_uniform_sampling,
            .smart_sampling_q = smart_sampling_q
            };

        agent =  new AUPO::AupoAgent(args);
    }

    if (agent != nullptr) {
        if (!extraArgs(agent_args, acceptable_args).empty()) {
            std::string err_string = "Invalid agent argument: " + extraArgs(agent_args, acceptable_args);
            std::cout << err_string << std::endl;
            throw std::runtime_error(err_string);
        }
        return agent;
    }else {
        std::cout << "Invalid agent" << std::endl;
        throw std::runtime_error("Invalid agent");
    }
}

int main(const int argc, char **argv) {


    argparse::ArgumentParser program("BenchmarkGames");

    program.add_argument("-s", "--seed")
        .help("Seed for the random number generator")
        .action([](const std::string &value) { return std::stoi(value); })
        .required();

    program.add_argument("-a", "--agent")
        .help("Agent to benchmark")
        .required();

    program.add_argument("--aargs")
        .help("Extra arguments for agent")
        .default_value(std::vector<std::string>{})
        .append();

    program.add_argument("-m", "--model")
        .help("Model to benchmark")
        .required();

    program.add_argument("--margs")
        .help("Extra arguments for model")
        .default_value(std::vector<std::string>{})
        .append();

    program.add_argument("-n", "--n_games")
        .help("Number of games to play")
        .action([](const std::string &value) { return std::stoi(value); })
        .required();

    program.add_argument("-v", "--csv")
        .help("CSV mode")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-p_horizon", "--p_horizon")
        .help("Planning horizon")
        .action([](const std::string &value) { return std::stoi(value); })
        .default_value(50);

    program.add_argument("-e_horizon", "--e_horizon")
    .help("Execution horizon")
    .action([](const std::string &value) { return std::stoi(value); })
    .default_value(100);

    program.add_argument("--planning_beyond_execution_horizon")
    .help("Whether the agent should plan beyond the execution horizon, i.e. always plan for the full planning horizon.")
    .default_value(false)
    .implicit_value(true);

    program.add_argument("--deterministic_init")
    .help("Whether to cycle through the same deterministic init states or sample random ones.")
    .default_value(false)
    .implicit_value(true);

    if (argc == 1) {
        std::cout << "Since no arguments were provided, for IDE convenience, the debug function will be called." << std::endl;
        debug();
        return 0;
    }

    program.parse_args(argc, argv);

    const auto seed = program.get<int>("--seed");
    std::mt19937 rng(seed);

    auto* model = getModel(program.get<std::string>("--model"), program.get<std::vector<std::string>>("--margs"));
    if (model == nullptr) {
        return 1;
    }

    std::vector<Agent*> agents;
    auto* agent = getAgent(program.get<std::string>("--agent"), program.get<std::vector<std::string>>("--aargs"));
    if (agent == nullptr) {
        delete model;
        return 1;
    }
    agents.push_back(agent);

    while (agents.size() < model->getNumPlayers()) {
        agents.push_back(getDefaultAgent());
    }

    auto horizons = std::make_pair(program.get<int>("--e_horizon"), program.get<int>("--p_horizon"));
    bool planning_beyond_execution_horizon = program.get<bool>("--planning_beyond_execution_horizon");
    bool random_init_state = !program.get<bool>("--deterministic_init");

    playGames(*model, program.get<int>("--n_games"), agents, rng, program.get<bool>("--csv") ? CSV: VERBOSE, horizons, planning_beyond_execution_horizon, random_init_state);

    delete model;
    for (auto agent_ptr : agents)
        delete agent_ptr;
    agents.clear();

    return 0;
}
