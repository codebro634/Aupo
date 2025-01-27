#pragma once

#ifndef GOL_H
#define GOL_H
#include <vector>

#include "../Gamestate.h"
#endif

namespace GOL
{

    struct Gamestate: public ABS::Gamestate{
        int board; // bits represent the alive/dead array

        bool operator==(const ABS::Gamestate& other) const override;
        [[nodiscard]] size_t hash() const override;
        [[nodiscard]] std::string toString() const override;
    };

    class Model: public ABS::Model
    {
    public:
        ~Model() override = default;
        explicit Model(const std::string& fileName);
        explicit Model();
        void printState(ABS::Gamestate* state) override;
        ABS::Gamestate* getInitialState(std::mt19937& rng) override;
        ABS::Gamestate* getInitialState(int num) override;
        ABS::Gamestate* copyState(ABS::Gamestate* uncasted_state) override;
        int getNumPlayers() override;

        [[nodiscard]] ABS::Gamestate* deserialize(std::string &ostring) const override;

    private:
        std::vector<std::vector<bool>> loaded_map;
        int loaded_map_hash;
        int map_width;
        std::vector<std::vector<float>> noise_map;
        std::pair<std::vector<double>,std::pair<int,double>> applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng) override;
        std::vector<int> getActions_(ABS::Gamestate* uncasted_state) override;

        [[nodiscard]] double getMinV(int steps) const override {return 0;}
        [[nodiscard]] double getMaxV(int steps) const override {return (map_width*map_width)*(getHorizonLength() - steps);}
        [[nodiscard]] double getDistance(const ABS::Gamestate* a, const ABS::Gamestate* b) const override;
    };

}

