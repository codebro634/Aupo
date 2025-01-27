#pragma once

#ifndef AUPOAGENT_H
#define AUPOAGENT_H
#include <functional>
#include <set>
#include "../Agent.h"
#include "AupoNode.h"
#include "../Mcts/MctsNode.h"

namespace AUPO {

    struct AupoBudget
    {
        int amount;
        std::string quantity;
    };

    struct AupoArgs
    {
        //standard mcts
        AupoBudget budget;
        std::vector<double> exploration_parameter = {2.0};
        double discount = 1.0;
        int rollout_length = -1; //-1 corresponds to rollout until terminal state

        //aupo specific
        int distribution_layers = 3;
        bool use_rollout_distribution = false;
        double confidence = 0.95;
        double confidence_std = -1;
        bool filter_by_std = false;
        bool dag = false;
        int min_samples = 10;
        bool smart_uniform_sampling = false;
        double smart_sampling_q = 0.9;

    };

    class AupoAgent final : public Agent
    {
    public:
        explicit AupoAgent(const AupoArgs& args);
        int getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng) override;

    private:

        int rollout_length;
        std::vector<double> exploration_parameter;
        double discount;
        AupoBudget budget;
        int distribution_layers;
        bool use_rollout_distribution;
        double confidence_mean;
        double confidence_std;
        bool dag;

        //For smart sampling only
        int min_samples;
        bool smart_uniform_sampling;
        double smart_sampling_q;
        constexpr static double TIEBREAKER_NOISE = 1e-6;

        struct AupoSearchStats
        {
            AupoBudget budget;
            int completed_iterations{};
            int total_forward_calls{};
            int max_depth{};
            std::map<int,Mcts::gsToNodeMap<AupoNode*>>* layerMap;

            //For global std exploration factor
            double total_squared_v = 0;
            double total_v = 0;
            int global_num_vs = 0;

            //Stats about the root node reward distribution
            std::map<std::pair<int,int>,std::pair<double,double>> mean_bound_cache; //cached confidence intervals for speedup
            std::map<std::pair<int,int>,std::pair<double,double>> std_bound_cache;
            std::map<std::pair<int,int>, double> rewards_per_layer;
            std::map<std::pair<int,int>, double> squared_rewards_per_layer;
            std::map<std::pair<int,int>,int> visits_per_layer;
        };

        std::vector<std::tuple<AupoNode*,int,double>> treePolicy(ABS::Model* model, AupoNode* node, std::mt19937& rng, AupoSearchStats& search_stats);
        AupoNode* selectNode(ABS::Model* model, AupoNode* node, bool& reached_leaf, int& chosen_action,double& reward, std::mt19937& rng, AupoSearchStats& search_stats);
        int decisionPolicy(AupoNode* root, std::mt19937& rng,  std::map<int,int>& abs_visits, std::map<int,double>& abs_value, std::map<int,std::set<int>>& abstracted_with, AupoSearchStats& search_stats);

        void filterBadActions(AupoNode *root, AupoSearchStats &search_stats, std::vector<int> &candidate_actions);

        int selectAction(AupoNode* node, std::mt19937& rng, AupoSearchStats& search_stats);
        std::pair<double,double> rollout(std::vector<double>& rewards, ABS::Model* model, AupoNode* node, std::mt19937& rng, AupoSearchStats& search_stats) const;
        void backup(std::vector<std::tuple<AupoNode*,int,double>>& path, std::vector<double>& rollout_rewards, double value, double squared_value, AupoSearchStats& search_stats);

        void update_abstraction(AupoNode* root, int action, std::map<int,int>& abs_visits, std::map<int,double>& abs_value, std::map<int,std::set<int>>& abstracted_with, AupoSearchStats& search_stats);

        //Downstream statistics for the root node
        [[nodiscard]] double getSampleVariance(int action, int layer, AupoSearchStats & stats);
        [[nodiscard]] std::pair<double,double> getQBounds(int action, double confidence, int min_samples, AupoSearchStats& stats, int parent_visits = -1);
        [[nodiscard]] double getSampleVariance(int action, AupoSearchStats& stats, int layer = -1);
        [[nodiscard]] std::pair<double,double> getStdBounds(int action, double confidence, int min_samples, AupoSearchStats& stats);
        [[nodiscard]] std::pair<double,double> getRewardMeanBounds(int action, int layer, double confidence, int min_samples, AupoSearchStats& stats, int parent_visits = -1);
        [[nodiscard]] std::pair<double,double> getRewardStdBounds(int action, int layer, double confidence, int min_samples, AupoSearchStats& stats);
        [[nodiscard]] int getLayerVisits(int action, int layer, AupoSearchStats& stats) {return stats.visits_per_layer.at({action,layer});}
        void addLayerVisitsAndReward(int action, int layer, double reward, AupoSearchStats& stats);
    };

}

#endif
