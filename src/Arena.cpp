#include <vector>
#include <iostream>
#include <cmath>
#include <map>
#include <algorithm>
#include "../include/Agents/Agent.h"
#include "../include/Arena.h"
#include <cassert>
#include <chrono>
#include <queue>
#include "../include/Utils/Distributions.h"

void outputStats(OutputMode& output_mode,
                 std::vector<std::pair<double,double>>& results,
                 std::vector<double>& regrets,
                 std::vector<std::pair<int,int>>& num_optimal_actions,
                 int games_played,
                 std::vector<int>& permutation,

                 std::vector<std::vector<int>>& played_actions,
                 std::vector<std::vector<double>>& individual_times,
                 std::vector<double>& reward_sum,
                 std::vector<unsigned>& num_actions,
                 std::vector<std::vector<double>>& regrets_last_game,
                 std::vector<double>& times)
{

    if (output_mode == VERBOSE) {
        std::cout << "------- Interim Results -------" << std::endl;
        for (int j = 0; j < results.size(); j++) {
            std::cout << "Player " << j << " total rewards: " << results[j].first << " in " << games_played << " games." << std::endl;
            if(num_optimal_actions[j].second > 0) {
                std::cout << "Player " << j << " ratio of optimal play: " << num_optimal_actions[j].first << "/" << num_optimal_actions[j].second << std::endl;
                std::cout << "Player " << j << " average regret: " << regrets[j] / num_optimal_actions[j].second << std::endl;
            }
            //for confidence interval
            int dofs = games_played - 1;
            const double confidence = 0.95;
            const double sample_variance = std::max(0.0,results[j].second/games_played - std::pow(results[j].first/games_played,2)) * games_played / (double)(games_played - 1);
            double studt_critical_value = distr::studt_quantile(1 - (1 -confidence) / 2, dofs, true);
            double conf_range = studt_critical_value * sqrt(sample_variance / games_played);
            std::cout << "Player " << j << " avg rewards: " << results[j].first/games_played << " +- " << conf_range <<  std::endl;
            std::cout << "Player " << j << " avg time: " << times[j]/num_actions[j] << std::endl;
            std::cout << "Player " << j << " played actions: ";
            for (int action_idx = 0; action_idx < played_actions[j].size(); action_idx++)
                std::cout << played_actions[j][action_idx] << " ";

            std::cout << std::endl;
        }

    }

    if (output_mode == CSV)
    {
        // Episode Nr.
        std::cout << games_played << ";";

        // Permutation
        for (int j : permutation)
        {
            std::cout << j << " ";
        }
        std::cout << ";";

        // Player info
        for (int j = 0; j < results.size(); j++)
        {
            std::cout << reward_sum[permutation[j]] << ";";

            for (double regret : regrets_last_game[permutation[j]])
            {
                std::cout << regret << " ";
            }
            std::cout << ";";

            for (const auto& played_action : played_actions[j])
            {
                std::cout << played_action << " ";
            }
            std::cout << ";";

            double sum_times = 0;
            for (const auto& individual_time : individual_times[j])
            {
                std::cout << individual_time << " ";
                sum_times += individual_time;
            }
            std::cout << ";";

            std::cout << sum_times;

            if (j != results.size() - 1)
            {
                std::cout << ";";
            }
        }

        std::cout << std::endl;
    }
}

void outputCsvHeader(int players)
{
    // Header for Episode Number
    std::cout << "Episode Nr;";

    // Header for Permutation
    std::cout << "Permutation;";

    // Headers for Player Info
    for (int j = 0; j < players; j++) {
        // Reward Sum for each player
        std::cout << "Rewards Player " << j << ";";

        // Cumulative regret for each player
        std::cout << "Regrets Player " << j << ";";

        // Played Actions for each player
        std::cout << "Actions Player " << j << ";";

        // Individual Times for each player
        std::cout << "Times Player " << j << ";";

        // Sum of Times for each player
        std::cout << "Total Time Player " << j;

        // Separate players by semicolons
        if (j != players-1) {
            std::cout << ";";
        }
    }

    // End the header row
    std::cout << std::endl;
}

std::vector<double> playGames(
    ABS::Model& model,
    int num_maps,
    std::vector<Agent*> agents,
    std::mt19937& rng,
    OutputMode output_mode,
    std::pair<int,int> horizons,
    bool planning_beyond_execution_horizon,
    bool random_init_state)
{
    std::vector<std::pair<double,double>> results = std::vector<std::pair<double,double>>(model.getNumPlayers(), {0,0}); //cumulative reward, and cumulative squared reward
    std::vector<double> regrets = std::vector<double>(model.getNumPlayers(), 0);
    std::vector<double> times = std::vector<double>(model.getNumPlayers(), 0);
    std::vector<unsigned> num_actions = std::vector<unsigned>(model.getNumPlayers(), 0);
    std::vector<std::pair<int,int>> num_optimal_action_chosen = std::vector<std::pair<int,int>>(model.getNumPlayers(), {0,0});

    assert (agents.size() == model.getNumPlayers());

    if (output_mode == CSV)
        outputCsvHeader(model.getNumPlayers());

    int games_played = 0;
    int num_perms = 1;
    for (int i = 2; i <= agents.size(); i++)
        num_perms *= i;

    for (int i = 0; i < num_maps; i++) {

        //iterate through all possible agent assignments
        std::vector<int> permutation(agents.size());
        for(int j = 0; j < agents.size(); j++)
            permutation[j] = j;

        do {
            auto played_actions = std::vector<std::vector<int>>(agents.size());
            auto individual_times = std::vector<std::vector<double>>(agents.size());

            ABS::Gamestate* gamestate = random_init_state ? model.getInitialState(rng) : model.getInitialState(i);
            std::vector<double> reward_sum = std::vector<double>(model.getNumPlayers(), 0);
            std::vector<std::pair<int,int>> num_optimal_action_chosen_last_game = std::vector<std::pair<int,int>>(model.getNumPlayers(), {0,0});
            std::vector<std::vector<double>> regrets_last_game = std::vector<std::vector<double>>(model.getNumPlayers(), std::vector<double>());

            while (!gamestate->terminal) {
                //choose action
                int planning_horizon = planning_beyond_execution_horizon? (horizons.second + gamestate->num_moves) :  std::min(horizons.first,horizons.second + gamestate->num_moves);
                const auto start = std::chrono::high_resolution_clock::now();
                model.setHorizonLength(planning_horizon); //planning horizon
                int action = agents[permutation[gamestate->turn]]->getAction(&model, gamestate, rng);
                model.setHorizonLength(horizons.first); //execution horizon

                //update statistics
                const auto time_elapsed = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
                times[permutation[gamestate->turn]] += time_elapsed;
                individual_times[permutation[gamestate->turn]].push_back(time_elapsed);
                num_actions[permutation[gamestate->turn]]++;
                auto key = std::make_pair(std::make_pair(gamestate,planning_horizon-gamestate->num_moves),action);

                //apply action
                auto rewards = model.applyAction(gamestate, action, rng).first;
                played_actions[permutation[gamestate->turn]].push_back(action);
                for (int j = 0; j < rewards.size(); j++)
                    reward_sum[j] += rewards[j];
            }

            games_played++;
            for(int k = 0; k < reward_sum.size(); k++) {
                results[k].first += reward_sum[permutation[k]];
                results[k].second += reward_sum[permutation[k]] * reward_sum[permutation[k]];
                for(double r : regrets_last_game[permutation[k]])
                    regrets[k] += r;
            }

            outputStats(output_mode, results, regrets, num_optimal_action_chosen, games_played, permutation, played_actions, individual_times, reward_sum, num_actions, regrets_last_game, times);

            delete gamestate;
        } while(std::next_permutation(permutation.begin(), permutation.end()));
     }

    auto cumulative_rewards = std::vector<double>(results.size());
    for (int i = 0; i < results.size(); i++)
        cumulative_rewards[i] = results[i].first;

    return cumulative_rewards;
}