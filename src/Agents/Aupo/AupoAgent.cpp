#include "../../../include/Agents/Aupo/AupoAgent.h"

#include <cassert>
#include <cmath>
#include <chrono>
#include <ranges>
#include "../../../include/Utils/Distributions.h"
#include <iostream>
#include <iomanip>

using namespace AUPO;

AupoAgent::AupoAgent(const AupoArgs& args) :
    rollout_length(args.rollout_length),
    exploration_parameter(args.exploration_parameter),
    discount(args.discount),
    budget(args.budget),
    distribution_layers(args.distribution_layers),
    use_rollout_distribution(args.use_rollout_distribution),
    dag(args.dag),
    min_samples(args.min_samples),
    smart_uniform_sampling(args.smart_uniform_sampling),
    smart_sampling_q(args.smart_sampling_q),
    random_abs_prob(args.random_abs_prob),
    just_mcts(args.just_mcts),
    distr_agent(args.distr_agent),
    earthmover_threshold(args.earthmover_threshold),
    ks_threshold(args.ks_threshold),
    asymptotic_std_ci(args.asymptotic_std_ci)
{
    confidence_mean = args.confidence;
    confidence_std = args.filter_by_std? (args.confidence_std == -1? args.confidence : args.confidence_std) : -1;
}

int AupoAgent::getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng)
{
    assert (model->getNumPlayers() == 1); // Only supports single player games

    const auto start = std::chrono::high_resolution_clock::now();
    auto init_state = model->copyState(state);
    auto* root = new AupoNode(model, init_state,0, rng);
    std::map<int,Mcts::gsToNodeMap<AupoNode*>> layerMap = {{0,{{init_state,root}}}};
    AupoSearchStats search_stats = {budget, 0, 0, 0,&layerMap,0,0,0,{},{}, {}, {}, {}, {}};
    const int total_forward_calls_before = model->getForwardCalls();

    while ( // Within budget
        (budget.quantity == "iterations" && search_stats.completed_iterations < budget.amount) ||
        (budget.quantity == "forward_calls" && search_stats.total_forward_calls < budget.amount) ||
        (budget.quantity == "milliseconds" && std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count() < budget.amount)
    )
    {
        // Aupo with forward calls cond can end in infinite loop
        auto leaf_path = treePolicy(model, root, rng, search_stats);
        auto rollout_rewards = std::vector<double>(distribution_layers, 0);
        const auto [val,squared_val] = rollout(rollout_rewards, model,std::get<0>(leaf_path.back()), rng, search_stats);
        backup(leaf_path,rollout_rewards, val,squared_val, search_stats);
        search_stats.completed_iterations++;
        search_stats.total_forward_calls = model->getForwardCalls() - total_forward_calls_before;
    }

    int best_action;
    if (!just_mcts){
        std::map<int,int>abs_visits;
        std::map<int,double> abs_value;
        std::map<int,std::set<int>> abstracted_with;
        for (auto& [layer, rewards] : search_stats.reward_history_per_layer)
            std::sort(rewards.begin(), rewards.end());
        for(int a : *root->getTriedActions())
            update_abstraction(root,a,abs_visits,abs_value,abstracted_with,search_stats,rng);
        best_action = decisionPolicy(root, rng, abs_visits, abs_value, abstracted_with, search_stats);
    }else{
        [[maybe_unused]] bool found = false;
        double best_val = std::numeric_limits<double>::lowest();
        auto dist = std::uniform_real_distribution<double>(-1.0, 1.0);
        for (int action : *root->getTriedActions()){
            double noise = TIEBREAKER_NOISE * dist(rng);
            const double q_term = root->getActionValues(action) / (double) root->getActionVisits(action);
            if(q_term + noise > best_val){
                best_val = q_term + noise;
                best_action = action;
                found = true;
            }
        }
        assert (found);
    }

    //tree cleanup
    std::vector<AupoNode*> clear_stack = {root};
    std::set<AupoNode*> to_clear_nodes = {root};
    while(!clear_stack.empty()){
        auto next = clear_stack.back();
        clear_stack.pop_back();
        for (auto& q_state : std::views::values(*next->getChildren())){
            for (const auto& child_state_node : std::views::values(q_state)){
                if(!to_clear_nodes.contains(child_state_node)){
                    clear_stack.push_back(child_state_node);
                    to_clear_nodes.insert(child_state_node);
                }
            }
        }
    }

    for(auto* node : to_clear_nodes){
        delete node->getState();
        delete node;
    }

    return distr_agent == nullptr? best_action : distr_agent->getAction(model, state, rng);
}

std::vector<std::tuple<AupoNode*,int,double>> AupoAgent::treePolicy(ABS::Model* model, AupoNode* node, std::mt19937& rng, AupoSearchStats& search_stats){
    bool reached_leaf = false;
    std::vector<std::tuple<AupoNode*,int,double>>  state_action_reward_path;
    auto old_node = node;
    while (!node->isTerminal() && !reached_leaf){
        int chosen_action;
        double reward;
        node = selectNode(model, node, reached_leaf, chosen_action, reward, rng, search_stats);
        state_action_reward_path.emplace_back(old_node,chosen_action, reward);
        old_node = node;
        if (node->getDepth() > search_stats.max_depth)
            search_stats.max_depth = node->getDepth();
    }
    state_action_reward_path.emplace_back(node,-1,0);
    return state_action_reward_path;
}

AupoNode* AupoAgent::selectNode(ABS::Model* model, AupoNode* node, bool& reached_leaf, int& chosen_action, double& reward,  std::mt19937& rng, AupoSearchStats& search_stats)
{
    reached_leaf = false;
    chosen_action = node->isFullyExpanded()? selectAction(node, rng, search_stats) : node->popUntriedAction();

    // Sample successor of state-action-pair
    auto sample_state = node->getStateCopy();
    auto [rewards_tmp, probability] = model->applyAction(sample_state, chosen_action, rng, nullptr);
    reward = rewards_tmp[0];

    const auto* successors = &node->getChildren()->at(chosen_action);
    if (!node->getChildren()->at(chosen_action).contains(sample_state))
    {
        // New successor sampled
        if( dag && (*search_stats.layerMap)[node->getDepth()+1].contains(sample_state))
        {
            auto* new_leaf = (*search_stats.layerMap)[node->getDepth()+1][sample_state];
            (*node->getChildren())[chosen_action][sample_state] = new_leaf;
            return new_leaf;
        }else{
            auto* new_leaf = new AupoNode(model, sample_state, node->getDepth() + 1, rng);
            reached_leaf = true;
            if(dag)
                (*search_stats.layerMap)[node->getDepth()+1][sample_state] = new_leaf;
            (*node->getChildren())[chosen_action][sample_state] = new_leaf;
            return new_leaf; //we dont delete sample state here because it has to be saved in the new node
        }
    }

    // Already sampled successor
    auto successor = successors->at(sample_state);
    delete sample_state;
    return successor;
}

void AupoAgent::filterBadActions(AupoNode *root, AupoSearchStats &search_stats, std::vector<int> &candidate_actions) {
    assert (root->getDepth() == 0);

    //get highest-lower confidence Q bound
    double best_lower_bound = std::numeric_limits<double>::lowest();
    std::map<int,double> upper_bounds = {};
    for(const int action : *root->getTriedActions()) {
        auto [mean_lower, mean_upper] = getQBounds(action, smart_sampling_q, min_samples,search_stats);
        best_lower_bound = std::max(best_lower_bound, mean_lower);
        upper_bounds[action] = mean_upper;
    }
    //add only those actions whose upper conf bound is bigger than the best lower bound
    for(const int action : *root->getTriedActions()) {
        double mean_upper = upper_bounds[action];
        if(mean_upper >= best_lower_bound)
            candidate_actions.push_back(action);
    }
}

int AupoAgent::decisionPolicy(AupoNode* root, std::mt19937& rng, std::map<int,int>& abs_visits, std::map<int,double>& abs_value, std::map<int,std::set<int>>& abstracted_with, AupoSearchStats &search_stats) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    assert (root->getDepth() == 0);

    //choose abstract action
    std::vector<int> actions;
    if(smart_uniform_sampling)
        filterBadActions(root, search_stats, actions);
    else
        actions = *root->getTriedActions();

    double best_abs_score = std::numeric_limits<double>::lowest();
    int best_abs_action = -42;
    for (const int action : actions){
        double score = abs_value[action] / (double) abs_visits[action] + TIEBREAKER_NOISE * dist(rng);
        if(score > best_abs_score) {
            best_abs_score = score;
            best_abs_action = action;
        }
    }

    // chose best ground action as max within best abs
    double best_score = std::numeric_limits<double>::lowest();
    int best_action = -42;
    for (const int action : abstracted_with[best_abs_action]){
        double score = root->getQ(action) + TIEBREAKER_NOISE * dist(rng);
        if(score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    return best_action;
}

int AupoAgent::selectAction(AupoNode* node, std::mt19937& rng, AupoSearchStats& search_stats)
{
    assert(!node->getTriedActions()->empty());

    //get exploration factor (global std)
    double exploration_param = node->getDepth() >= static_cast<int>(exploration_parameter.size()) ? exploration_parameter.back() : exploration_parameter[node->getDepth()];
    const double q_var = std::max(0.0,search_stats.total_squared_v / (double)search_stats.global_num_vs - (search_stats.total_v / (double)search_stats.global_num_vs) *  (search_stats.total_v / (double)search_stats.global_num_vs));
    double exp_factor = exploration_param == -1? 1.0 : (sqrt(q_var) *  exploration_param);

    //determine abstract action with UCT
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<int> actions;
    if(smart_uniform_sampling && exploration_param == -1)
        filterBadActions(node, search_stats, actions);
    else
        actions = *node->getTriedActions();
    double best_score = std::numeric_limits<double>::lowest();
    int best_action = -42;
    for (const int action : actions){
        double noise = TIEBREAKER_NOISE * dist(rng);
        const double exploration_term = sqrt(log(node->getVisits()) / (node->getActionVisits(action)));
        const double q_term = exploration_param == -1? 0.0 : (node->getActionValues(action) / (double) node->getActionVisits(action));
        double score = q_term + exp_factor * exploration_term + noise;
        if(score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    return best_action;
}

std::pair<double,double> AupoAgent::rollout(std::vector<double>& rewards, ABS::Model* model, AupoNode* node, std::mt19937& rng, AupoSearchStats& search_stats) const{

    double reward_sum = 0, squared_reward_sum = 0;
    if (node->isTerminal())
        return {reward_sum, squared_reward_sum};

    double total_discount = discount;
    auto* rollout_state = node->getStateCopy();
    double episode_rew = 0;
    int episode_step = 0;
    while (!rollout_state->terminal && (rollout_length == -1 || episode_step < rollout_length))
    {
        // Sample action
        auto available_actions = model->getActions(rollout_state);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(available_actions.size()) - 1);
        const int action = available_actions[dist(rng)];

        // Apply action and get rewards
        auto [reward, outcome_and_probability] = model->applyAction(rollout_state, action, rng, nullptr);
        episode_rew += reward[0] * total_discount;
        if(episode_step < distribution_layers)
            rewards[episode_step] = reward[0];
        total_discount *= discount;
        episode_step++;
    }
    delete rollout_state;

    reward_sum += episode_rew;
    squared_reward_sum += episode_rew * episode_rew;

    return {reward_sum, squared_reward_sum};
}

void AupoAgent::backup(std::vector<std::tuple<AupoNode*,int,double>>& path, std::vector<double>& rollout_rewards, double value, double squared_value, AupoSearchStats& search_stats)
{
    std::map<int,double> rewards_per_layer;
    int max_depth = std::get<0>(path[path.size()-1])->getDepth()-1;

    for (int i = path.size()-2; i >= 0; i--)
    {
        auto node = std::get<0>(path[i]);
        auto action = std::get<1>(path[i]);
        auto reward = std::get<2>(path[i]);

        squared_value = reward * reward + discount * discount * squared_value + 2 * discount * value * reward;
        value = value * discount + reward;

        rewards_per_layer[node->getDepth()] = reward;
        node->addVisits(1);
        node->addActionVisits(action, 1);
        node->addActionValues(action, value);

        //global std exploration factor bookkeeping
        if(node->getActionVisits(action) == 1)
            search_stats.global_num_vs++;
        if(node->getActionVisits(action) > 1) { //only remove value if it was present before
            double old_q = (node->getActionValues(action)-value) / ((double) node->getActionVisits(action)-1);
            search_stats.total_v -= old_q;
            search_stats.total_squared_v -= old_q * old_q;
        }
        double q = node->getActionValues(action) / (double) node->getActionVisits(action);
        search_stats.total_v += q;
        search_stats.total_squared_v += q*q;

        if (!just_mcts){
            search_stats.squared_rewards_per_layer[{action,-1}] += squared_value;
            search_stats.rewards_per_layer[{action,-1}] += value;
            search_stats.visits_per_layer[{action,-1}]++;
            search_stats.mean_bound_cache.erase({action,-1});
            search_stats.std_bound_cache.erase({action,-1});
            search_stats.reward_history_per_layer[{action,-1}].push_back(value);

            if(node->getDepth() == 0){
                for(int i = 0; i < distribution_layers; i++){
                    auto downstream_reward =  i + node->getDepth() <= max_depth? rewards_per_layer.at(i+node->getDepth()) : rollout_rewards[i + node->getDepth() - max_depth - 1];
                    addLayerVisitsAndReward(action,i,downstream_reward,search_stats);
                }
            }
        }
    }

}

void AupoAgent::update_abstraction(AupoNode* root, int action, std::map<int,int>& abs_visits, std::map<int,double>& abs_value, std::map<int,std::set<int>>& abstracted_with, AupoSearchStats& search_stats, std::mt19937& rng) {
    assert (root->getDepth() == 0);
    abs_value[action] = 0;
    abs_visits[action] = 0;
    for(auto in : abstracted_with[action]) {
        if(in == action)
            continue;
        abs_visits[in] -= root->getActionVisits(action);
        abs_value[in] -= root->getActionValues(action);
        abstracted_with[in].erase(action);
    }
    abstracted_with[action].clear();

    //Setup confidence interval bounds
    std::vector<std::pair<double,double>> node_mean_bounds = std::vector<std::pair<double,double>>(distribution_layers, {0,0});
    std::vector<std::pair<double,double>> node_std_bounds = std::vector<std::pair<double,double>>(distribution_layers, {0,0});
    for(int i = 0; i < distribution_layers; i++) {
        node_mean_bounds[i] = getRewardMeanBounds(action, i, confidence_mean, min_samples,search_stats);
        node_std_bounds[i] = confidence_std > 0? getRewardStdBounds(action, i, confidence_std, min_samples,search_stats) : std::pair<double,double>(0,0);
    }
    std::pair<double,double> rollout_mean_bounds, rollout_std_bounds;
    if(use_rollout_distribution) {
        rollout_mean_bounds = getQBounds(action, confidence_mean, min_samples,search_stats);
        rollout_std_bounds = confidence_std > 0? getStdBounds(action, confidence_std, min_samples,search_stats) : std::pair<double,double>(0,0);
    }


    Mcts::gsToNodeMap<AupoNode*> this_pair = {{nullptr,root}};
    for(int other_action:  *root->getTriedActions()) {
        bool separated = false;

        for(int i = use_rollout_distribution? -1: 0; i < distribution_layers; i++) {
            if (ks_threshold != -1) {
                auto& reward_history_this = search_stats.reward_history_per_layer.at({action,i});
                auto& reward_history_other = search_stats.reward_history_per_layer.at({other_action,i});
                const std::size_t n = reward_history_this.size();
                const std::size_t m = reward_history_other.size();

                double d=0;
                if (n == 0 && m == 0) d = 0;
                else if (n == 0 || m == 0) d = 1;
                else {
                    std::size_t i = 0; // index into a
                    std::size_t j = 0; // index into b

                    while (i < n || j < m) {
                        double x;
                        if (i < n && (j == m || reward_history_this[i] < reward_history_other[j]))
                            x = reward_history_this[i];
                        else if (j < m && (i == n || reward_history_other[j] < reward_history_this[i]))
                            x = reward_history_other[j];
                        else
                            x = reward_history_this[i];

                        while (i < n && std::fabs(reward_history_this[i] - x) < TIEBREAKER_NOISE) ++i;
                        while (j < m && std::fabs(reward_history_other[j] - x) < TIEBREAKER_NOISE) ++j;

                        double Fa = static_cast<double>(i) / static_cast<double>(n);
                        double Fb = static_cast<double>(j) / static_cast<double>(m);

                        d = std::max(d, std::fabs(Fa - Fb));
                    }
                }

                if (d > ks_threshold) {
                    separated = true;
                    break;
                }

            }
            else if (earthmover_threshold != -1) {
                double wasserstein_dist = 0;
                auto& reward_history_this = search_stats.reward_history_per_layer.at({action,i});
                auto& reward_history_other = search_stats.reward_history_per_layer.at({other_action,i});

                for (size_t i = 0; i < std::min(reward_history_this.size(), reward_history_other.size()); i++)
                    wasserstein_dist += std::abs((double)reward_history_this[i] - (double)reward_history_other[i]);
                wasserstein_dist /= (double) std::min(reward_history_this.size(), reward_history_other.size());
                if (wasserstein_dist > earthmover_threshold) {
                    separated = true;
                    break;
                }
            }
            else{
                auto [node_mean_lower, node_mean_upper] = i == -1? rollout_mean_bounds : node_mean_bounds[i];
                auto [node_std_lower, node_std_upper] = i == -1? rollout_std_bounds : node_std_bounds[i];
                node_mean_lower -= TIEBREAKER_NOISE;
                node_mean_upper += TIEBREAKER_NOISE;
                node_std_lower -= TIEBREAKER_NOISE;
                node_std_upper += TIEBREAKER_NOISE;
                auto [other_mean_lower, other_mean_upper] = getRewardMeanBounds(other_action, i, confidence_mean, min_samples,search_stats);
                auto [other_std_lower, other_std_upper] = confidence_std > 0? getRewardStdBounds(other_action, i, confidence_std, min_samples,search_stats) : std::pair<double,double>(0,0);
                bool mean_cond = (node_mean_lower <= other_mean_upper && node_mean_upper >= other_mean_upper) ||
                    (node_mean_lower <= other_mean_lower && node_mean_upper >= other_mean_lower)
                    || (other_mean_lower <= node_mean_lower &&other_mean_upper >= node_mean_lower);
                bool std_cond = confidence_std < 0 || (node_std_lower >= other_std_lower && node_std_lower <= other_std_upper) ||
                    (node_std_upper >= other_std_lower && node_std_upper <= other_std_upper)
                || (node_std_lower <= other_std_lower && node_std_upper >= other_std_lower);
                if(!mean_cond || !std_cond) {
                    separated = true;
                    break;
                }
            }
        }

        if((!separated && random_abs_prob == 0) || (random_abs_prob > 0 && (other_action == action || std::bernoulli_distribution(random_abs_prob)(rng)))) {
            abs_value[action] += root->getActionValues(other_action);
            abs_visits[action] += root->getActionVisits(other_action);
            abstracted_with[action].insert(other_action);

            if(other_action != action) {
                abs_visits[other_action] += root->getActionVisits(action);
                abs_value[other_action] += root->getActionValues(action);
                abstracted_with[other_action].insert(action);
            }
        }

    }

    assert(abstracted_with[action].contains(action));

}

double AupoAgent::getSampleVariance(int action, int layer, AupoAgent::AupoSearchStats & stats) {
    double visits = stats.visits_per_layer.at({action,layer});
    assert (visits > 0);
    if(visits == 1)
        return 0;

    double squared_val = stats.squared_rewards_per_layer.at({action,layer});
    double correction_factor = visits / (double)(visits - 1);
    double Q = stats.rewards_per_layer.at({action,layer}) / (double)visits;

    return std::max(0.0,correction_factor * (squared_val / visits - pow(Q,2)));
}

void AupoAgent::addLayerVisitsAndReward(int action, int layer, double reward, AupoSearchStats & stats) {
    stats.visits_per_layer[{action,layer}]++;
    stats.rewards_per_layer[{action,layer}] += reward;
    stats.squared_rewards_per_layer[{action,layer}] += reward * reward;
    stats.mean_bound_cache.erase({action,layer});
    stats.std_bound_cache.erase({action,layer});
    stats.reward_history_per_layer[{action,layer}].push_back(reward);
}

std::pair<double,double> AupoAgent::getQBounds(const int action, const double confidence, int min_samples, AupoAgent::AupoSearchStats & stats, int parent_visits){
    return getRewardMeanBounds(action,-1,confidence,min_samples,stats, parent_visits);
}

std::pair<double, double> AupoAgent::getRewardMeanBounds(int action, int layer, double confidence, int min_samples, AupoAgent::AupoSearchStats & stats, int parent_visits)
{
    // if(stats.mean_bound_cache.contains({action,layer}) && parent_visits == -1)
    //     return stats.mean_bound_cache.at({action,layer});

    std::pair<double,double> bounds;
    if(!stats.visits_per_layer.contains({action,layer}) || stats.visits_per_layer.at({action,layer}) < min_samples)
        bounds = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
    else {
        int layer_visits = stats.visits_per_layer.at({action,layer});
        double std_err = sqrt(getSampleVariance(action,layer,stats) / (double) layer_visits);
        int df = layer_visits - 1;
        double err = std_err * (confidence > 1? confidence : distr::studt_quantile(1 - (1 -confidence) / 2, df, true));
        err *= parent_visits == -1? 1 : std::max(1.0,std::sqrt(std::log(parent_visits)));
        double Q = stats.rewards_per_layer.at({action,layer}) / (double)layer_visits;
        bounds = {Q - err, Q + err};
    }

    if(parent_visits == -1)
        stats.mean_bound_cache[{action,layer}] = bounds;

    return bounds;
}


std::pair<double,double> AupoAgent::getStdBounds(const int action, const double confidence, int min_samples, AupoAgent::AupoSearchStats & stats){
    return getRewardStdBounds(action,-1,confidence, min_samples,stats);
}

std::pair<double, double> AupoAgent::getRewardStdBounds(int action, int layer, double confidence, int min_samples, AupoAgent::AupoSearchStats & stats)
{
    assert (confidence > 0 && confidence < 1);

    // if(stats.std_bound_cache.contains({action,layer}))
    //     return stats.std_bound_cache.at({action,layer});

    std::pair<double,double> bounds;
    if(!stats.visits_per_layer.contains({action,layer}) || stats.visits_per_layer.at({action,layer}) < min_samples)
        bounds = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
    else if (asymptotic_std_ci) {
        int layer_visits = stats.visits_per_layer.at({action,layer});
        double alpha = 1.0 - confidence;
        double variance = getSampleVariance(action,layer,stats);
        double z = confidence > 1? confidence : distr::normal_quantile(1 - alpha / 2);

        double m2 = variance * (double)(layer_visits - 1) / (double)layer_visits;
        if (m2 == 0)
            return {0,0};

        std::vector<double>& rewards = stats.reward_history_per_layer.at({action,layer});
        double mean = stats.rewards_per_layer.at({action,layer}) / (double) layer_visits;
        double m4 = 0;
        for(double r : rewards)
            m4 += pow(r - mean, 4);
        m4 /= (double) layer_visits;

        double kurtosis = m4 / (m2 * m2);

        double lowerBound = sqrt(variance) - z*sqrt((kurtosis - 1) * variance  / (4.0 * (double) layer_visits));
        double upperBound = sqrt(variance) + z*sqrt((kurtosis - 1) * variance / (4.0 * (double) layer_visits));

        bounds = {std::max(0.0,lowerBound), std::max(0.0,upperBound)};
    }
    else {
        int layer_visits = stats.visits_per_layer.at({action,layer});
        double alpha = 1.0 - confidence;
        double variance = getSampleVariance(action,layer,stats);

        // Degrees of freedom
        int df = layer_visits - 1;

        // Critical values for the Chi-squared distribution
        double chi2Lower = distr::chi2_quantile(alpha / 2, df, true);
        double chi2Upper = distr::chi2_quantile(1 - alpha / 2, df, true);

        // Confidence interval for standard deviation
        double lowerBound = sqrt((df * variance) / chi2Upper);
        double upperBound = sqrt((df * variance) / chi2Lower);

        bounds = {lowerBound, upperBound};
    }

    stats.std_bound_cache[{action,layer}] = bounds;

    return bounds;
}