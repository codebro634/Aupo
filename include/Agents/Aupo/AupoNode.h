#pragma once

#ifndef AupoNODE_H
#define AupoNODE_H
#include <map>
#include <random>
#include <unordered_set>
#include <vector>

#include "../../Arena.h"
#endif

namespace AUPO {

    inline long global_id = 0;

    class AupoNode
    {
    public:
        AupoNode(
            ABS::Model* model,
            ABS::Gamestate* state,
            int depth,
            std::mt19937& rng
        );

        int popUntriedAction();

        void addVisits(int visits);
        void addActionVisits(int action, int visits);
        void addActionValues(int action, double value);

        [[nodiscard]] double getActionValues(int action, bool init_default = false);
        [[nodiscard]] double getQ(int action) {return getActionValues(action) / getActionVisits(action);}


        [[nodiscard]] ABS::Model* getModel() const;
        [[nodiscard]] ABS::Gamestate* getStateCopy() const;
        [[nodiscard]] ABS::Gamestate* getState() const {return state;}
        [[nodiscard]] std::map<int, std::map<int, AupoNode*>>* getChildren();
        [[nodiscard]] int getPlayer() const;


        [[nodiscard]] int getDepth() const;
        [[nodiscard]] int getVisits() const;
        [[nodiscard]] int getActionVisits(int action, bool init_default = false);
        [[nodiscard]] bool isFullyExpanded() const;
        [[nodiscard]] bool isTerminal() const;
        [[nodiscard]] std::vector<int>* getTriedActions();

        [[nodiscard]] std::string toString();

        ~AupoNode() = default;

        long id = global_id++;

    private:
        // Model related stats
        ABS::Model* model;
        ABS::Gamestate* state;
        std::map<int, std::map<int, AupoNode*>> children;
        std::map<int, int> action_visits;
        std::map<int, double> action_values;
        int depth;
        int visits;
        std::vector<int> tried_actions;
        std::vector<int> untried_actions;
    };
}
