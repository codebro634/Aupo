#pragma once

#ifndef GAMESTATECONTROLLER_H
#define GAMESTATECONTROLLER_H
#include <cassert>
#include <vector>
#include <iostream>
#include <random>
#endif

namespace ABS
{

    struct Gamestate{
        int num_moves=0;
        int turn=0; //player whose turn it is
        bool terminal=false;
        virtual ~Gamestate() = default;

        [[nodiscard]] virtual std::string toString() const {
            return "(" + std::to_string(num_moves) + "," + std::to_string(turn) + "," + std::to_string(terminal) + ")";
        }

        friend std::ostream& operator<<(std::ostream& os, const Gamestate& state) {
            // Dynamically execute the function that converts the state to a string
            os << state.toString();
            return os;
        }

        // Functions for hashing and comparison needed for unordered_map and unordered_set
        virtual bool operator==(const Gamestate& other) const
        {
            throw std::runtime_error("Equality not implemented");
        }
        [[nodiscard]] virtual size_t hash() const
        {
            throw std::runtime_error("Hash not implemented");
        }

    };

    class Model {

        protected:
            virtual std::pair<std::vector<double>,std::pair<int,double>> applyAction_(Gamestate* uncasted_state, int action, std::mt19937& rng)=0;
            virtual std::vector<int> getActions_(Gamestate* uncasted_state)=0;
            long total_forward_calls = 0;
            int horizon_length = -1;

        public:
            virtual ~Model() = default;
            virtual void printState(Gamestate* uncasted_state)=0;

            virtual Gamestate* getInitialState(std::mt19937& rng)=0; //Random init state
            virtual Gamestate* getInitialState(int num){ throw std::runtime_error("Deterministic initial state not implemented.");} //Deterministic init state
            virtual int getNumPlayers()=0;
            virtual Gamestate* copyState(Gamestate* uncasted_state)=0;

            virtual void setHorizonLength(int horizon_length_){
                horizon_length = horizon_length_;
            }

            [[nodiscard]] virtual int getHorizonLength() const{
                return horizon_length;
            }

            virtual std::vector<int> getActions(Gamestate* uncasted_state) final {
                assert (!uncasted_state->terminal); //state must not be terminal
                return getActions_(uncasted_state);
            };

            //Assertions:
            //1. Reward depends only on the state and action and NOT the sampled successor, i.e. R(s,a)
            //2. Two sampled successor states of the same state-action pair are equal only iff their outcome numbers are the same
            virtual std::pair<std::vector<double>,std::pair<int,double>> applyAction(Gamestate* uncasted_state, int action, std::mt19937& rng) final {
                assert (horizon_length != -1); //horizon length must be set
                assert (!uncasted_state->terminal); //state must not be terminal
                total_forward_calls++;
                uncasted_state->num_moves++;
                auto result = applyAction_(uncasted_state, action, rng);
                if (uncasted_state->num_moves >= horizon_length)
                    uncasted_state->terminal = true;
                return result;
            }; //return value is reward + number of sample + probability of sample.


            virtual long getForwardCalls() {
                return total_forward_calls;
            };

            //Functions for algorithms that need a heuristic value of the state
            [[nodiscard]] virtual double heuristicsValue(Gamestate* uncasted_state) const {
                throw std::runtime_error("Heuristics not implemented.");
            }

            [[nodiscard]] virtual double getMaxV(int steps) const {
                throw std::runtime_error("MaxV not implemented.");
            }

            [[nodiscard]] virtual double getMinV(int steps) const {
                throw std::runtime_error("MinV not implemented.");
            }

            [[nodiscard]] virtual double getDistance(const Gamestate* a, const Gamestate* b) const {
                throw std::runtime_error("Distance not implemented.");
            }

            //Deserialization
            [[nodiscard]] virtual ABS::Gamestate* deserialize(std::string& ostring) const {
                throw std::runtime_error("Deserialization not implemented.");
            }


    };
}
