#pragma once

#ifndef ARENA_H
#define ARENA_H

#include <vector>
#include <random>
#include <set>
#include <unordered_map>

#include "Games/Gamestate.h"
#include "Agents/Agent.h"

#endif //ARENA_H

static constexpr int DEFAULT_EXECUTION_HORIZON = 50;
static constexpr int DEFAULT_PLANNING_HORIZON = 50;

enum OutputMode
{
    NORMAL,
    VERBOSE,
    CSV
};

std::vector<double> playGames(ABS::Model& model,
    int num_maps,
    std::vector<Agent*> agents,
    std::mt19937& rng,
    OutputMode output_mode,
    std::pair<int,int> horizons = {DEFAULT_EXECUTION_HORIZON,DEFAULT_PLANNING_HORIZON},
    bool planning_beyond_execution_horizon = false,
    bool random_init_state =true);
