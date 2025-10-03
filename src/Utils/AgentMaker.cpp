#include "../../include/Utils/AgentMaker.h"
#include "../../include/Agents/Mcts/MctsAgent.h"
#include "../../include/Agents/Aupo/AupoAgent.h"
#include "../../include/Games/TwoPlayerGames/Pusher.h"
#include "../../include/Agents/RandomAgent.h"
#include "../../include/Agents/HumanAgent.h"
#include "../../include/Agents/SparseSamplingAgent.h"

#include <map>
#include <set>
#include <sstream>


Agent* getDefaultAgent(bool strong){
    if (strong)
        return new Mcts::MctsAgent({{500, "iterations"}, {4}, 1.0, 1, -1, true, true});
    else
       return new RandomAgent();
}

std::string extraArgs(std::map<std::string, std::string>& given_args, const std::set<std::string>& acceptable_args){
    for (auto& [key, val] : given_args) {
        if(!acceptable_args.contains(key))
            return key;
    }
    return "";
}

Agent* getAgent(const std::string& agent_type, const std::vector<std::string>& a_args)
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
    if (agent_type == "random") {
        acceptable_args = {};
        agent =  new RandomAgent();
    }
    else if(agent_type == "mcts")
    {
        assert (agent_args.contains("iterations"));
        if(agent_args.contains("wirsa"))
            assert (agent_args.contains("a") && agent_args.contains("b"));
        acceptable_args = {"iterations", "rollout_length", "discount", "num_rollouts", "dag", "dynamic_exp_factor", "expfacs", "wirsa", "a", "b", "puct", "max_backup"};

        int iterations = std::stoi(agent_args["iterations"]);
        int rollout_length = agent_args.find("rollout_length") == agent_args.end() ? -1 : std::stoi(agent_args["rollout_length"]);
        double discount = agent_args.find("discount") == agent_args.end() ? 1.0 : std::stod(agent_args["discount"]);
        int num_rollouts = agent_args.find("num_rollouts") == agent_args.end() ? 1: std::stoi(agent_args["num_rollouts"]);
        bool dag = agent_args.find("dag") == agent_args.end() ? false : std::stoi(agent_args["dag"]);
        bool dynamic_exp_factor = agent_args.find("dynamic_exp_factor") == agent_args.end() ? false : std::stoi(agent_args["dynamic_exp_factor"]);
        bool wirsa = agent_args.find("wirsa") == agent_args.end() ? false : std::stoi(agent_args["wirsa"]);
        double a = agent_args.find("a") == agent_args.end() ? 0.0 : std::stod(agent_args["a"]);
        double b = agent_args.find("b") == agent_args.end() ? 0.0 : std::stod(agent_args["b"]);
        std::string exp_facs = agent_args.find("expfacs") == agent_args.end() ? "1" : agent_args["expfacs"];
        std::vector<double> expfac;
        std::stringstream ss(exp_facs);
        double i;
        while (ss >> i){
            expfac.push_back(i);
            if (ss.peek() == ';')
                ss.ignore();
        }
        bool max_backup = agent_args.find("max_backup") == agent_args.end() ? false : std::stoi(agent_args["max_backup"]);
        bool puct = agent_args.find("puct") == agent_args.end() ? false : std::stoi(agent_args["puct"]);

        auto args = Mcts::MctsArgs{.budget = {iterations, "iterations"}, .exploration_parameters = expfac, .discount = discount,
            .num_rollouts = num_rollouts,
            .rollout_length = rollout_length,
            .dag=dag,
            .dynamic_exploration_factor=dynamic_exp_factor,
            .max_backup = max_backup,
            .wirsa = wirsa,
            .a=a,.b=b, .puct = puct};
        agent =  new Mcts::MctsAgent(args);
    }
     else if (agent_type == "aupo") {

        assert (agent_args.contains("iterations"));
        acceptable_args = {"iterations", "discount", "expfacs", "rollout_length", "distribution_layers", "confidence", "confidence_std",
            "filter_by_std", "use_rollout_distribution", "K", "dag", "min_samples", "smart_uniform_sampling", "smart_sampling_q", "random_abs_prob",
            "earthmover_threshold", "ks_threshold", "asymptotic_std_ci"};

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
        int min_samples = agent_args.find("min_samples") == agent_args.end() ? 10 : std::stoi(agent_args["min_samples"]);
        bool smart_uniform_sampling = agent_args.find("smart_uniform_sampling") == agent_args.end() ? false : std::stoi(agent_args["smart_uniform_sampling"]);
        double smart_sampling_q = agent_args.find("smart_sampling_q") == agent_args.end() ? 0.9 : std::stod(agent_args["smart_sampling_q"]);
        double random_abs_prob = agent_args.find("random_abs_prob") == agent_args.end() ? 0.0 : std::stod(agent_args["random_abs_prob"]);
        double earthmover_threshold = agent_args.find("earthmover_threshold") == agent_args.end() ? -1 : std::stod(agent_args["earthmover_threshold"]);
        double ks_threshold = agent_args.find("ks_threshold") == agent_args.end() ? -1 : std::stod(agent_args["ks_threshold"]);
        bool asymptotic_std_ci = agent_args.find("asymptotic_std_ci") == agent_args.end() ? false : std::stoi(agent_args["asymptotic_std_ci"]);

        auto args = AUPO::AupoArgs{.budget = {iterations, "iterations"}, .exploration_parameter = expfac,
            .discount = discount, .rollout_length = rollout_length,
            .distribution_layers = distribution_layers, .use_rollout_distribution = use_rollout_distribution,
            .confidence = confidence, .confidence_std = confidence_std,
            .filter_by_std = filter_by_std,
            .dag = dag,
            .min_samples = min_samples,
            .smart_uniform_sampling = smart_uniform_sampling,
            .smart_sampling_q = smart_sampling_q,
            .random_abs_prob = random_abs_prob,
            .earthmover_threshold = earthmover_threshold,
            .ks_threshold = ks_threshold,
            .asymptotic_std_ci = asymptotic_std_ci,
            };

        agent =  new AUPO::AupoAgent(args);
    }else{
        throw std::runtime_error("Invalid agent");
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