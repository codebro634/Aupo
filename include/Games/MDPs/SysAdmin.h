#pragma once

#ifndef SA_H
#define SA_H
#include <vector>

#include "../Gamestate.h"
#endif

namespace SA
{

    const static float REBOOT_PROB = 0.05;
    const static float REBOOT_COST = 0.75;

    const static std::vector<std::vector<int>> DEFAULT_CONNECTIONS = {
        {0, 1, 1, 0, 0},
        {1, 0, 1, 1, 0},
        {1, 1, 0, 1, 1},
        {0, 1, 1, 0, 1},
        {0, 0, 1, 1, 0}
    };

    struct Gamestate: public ABS::Gamestate{
        int machine_statuses;
        bool operator==(const ABS::Gamestate& other) const override;
        [[nodiscard]] size_t hash() const override;
    };

    class Model: public ABS::Model
    {
    public:
        ~Model() override = default;
        explicit Model(const std::string& fileName, bool deterministic = false);
        explicit Model(bool deterministic = false);
        void printState(ABS::Gamestate* state) override;
        ABS::Gamestate* getInitialState(std::mt19937& rng) override;
        ABS::Gamestate* getInitialState(int num) override;
        ABS::Gamestate* copyState(ABS::Gamestate* uncasted_state) override;
        int getNumPlayers() override;

        [[nodiscard]] double getMinV(int steps) const override {return 0;}
        [[nodiscard]] double getMaxV(int steps) const override {return (connections.size())*(getHorizonLength() - steps);}
        [[nodiscard]] double getDistance(const ABS::Gamestate* a, const ABS::Gamestate* b) const override;


    private:
        bool deterministic;
        std::vector<std::vector<int>> connections;
        std::vector<int> actions;
        std::pair<std::vector<double>,std::pair<int,double>> applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng) override;
        std::vector<int> getActions_(ABS::Gamestate* uncasted_state) override;
    };

}

