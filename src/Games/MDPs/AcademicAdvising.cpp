#include <bits/atomic_base.h>

#include "../../../include/Games/MDPs/AcademicAdvising.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
using namespace std;

using namespace AA;


[[nodiscard]] std::string Gamestate::toString() const {
    return "((" + std::to_string(passed_courses) + "," + std::to_string(taken_courses) + ")" + "," + ABS::Gamestate::toString() + ")";
}


ABS::Gamestate* Model::deserialize(std::string &ostring) const {
    auto* state = new Gamestate();
    int passed, taken, turn, num_moves, terminal;
    sscanf(ostring.c_str(), "((%d,%d),(%d,%d,%d))", &passed, &taken, &num_moves, &turn, &terminal);
    state->passed_courses = passed;
    state->taken_courses = taken;
    state->num_taken = 0;
    state->num_passed = 0;

    auto req_set = std::set<int>(req_courses.begin(), req_courses.end());

    for(int i = 0; i < prereqs.size(); i++){
        if(!state->isIthCoursePassed(i) && req_set.contains(i))
                state->missing_reqs.insert(i);
        if(state->isIthCoursePassed(i))
            state->num_passed++;
        if(state->isIthCourseTaken(i))
            state->num_taken;
    }

    state->num_moves = num_moves;
    state->turn = turn;
    state->terminal = terminal;
    return state;
}

Model::Model(const std::string& fileName, bool dense_rewards)
{
    this->dense_rewards = dense_rewards;
    //std::cout << "Reading file " << fileName << std::endl;
    
    std::ifstream file(fileName); // Open the file
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << fileName << std::endl;
        return;
    }

    std::string line;
    bool first_line = true;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream iss(line); // Create a string stream for each line
        std::vector<int> courses;
        int course;

        // Parse integers from the line
        while (iss >> course) {
            courses.push_back(course);
        }

        // If it's the first line, populate req_courses
        if (first_line) {
            req_courses = courses; // Transfer parsed data to req_courses
            first_line = false;
        } else {
            prereqs.push_back(courses); // Add parsed data to prereqs
        }
    }
    for(int i = 0; i < prereqs.size(); i++)
        actions.push_back(i);
    file.close(); // Close the file
    assert (prereqs.size() <= 32); //We are using a 32 bit integer to store the passed courses. If one wants to increase this limit, change == operator
}

inline bool Gamestate::isIthCoursePassed(int i)
{
    return passed_courses & (1 << i);
}

inline bool Gamestate::isIthCourseTaken(int i)
{
    return taken_courses & (1 << i);
}

inline void Gamestate::setIthCoursePassed(int i)
{
    passed_courses |= (1 << i);
}

inline void Gamestate::setIthCourseTaken(int i)
{
    taken_courses |= (1 << i);
}

double Model::getDistance(const ABS::Gamestate* a, const ABS::Gamestate* b) const {
    const Gamestate* state_a = (Gamestate*) a;
    const Gamestate* state_b = (Gamestate*) b;
    return __builtin_popcount( state_a->passed_courses ^ state_b->passed_courses) + __builtin_popcount( state_a->taken_courses ^ state_b->taken_courses);
}

bool Gamestate::operator==(const ABS::Gamestate& other) const
{
    return passed_courses == dynamic_cast<const Gamestate&>(other).passed_courses && taken_courses == dynamic_cast<const Gamestate&>(other).taken_courses;
}

size_t Gamestate::hash() const
{
    return passed_courses | (taken_courses << 16);
}

void Model::printState(ABS::Gamestate* state) {
    auto* AAState = dynamic_cast<AA::Gamestate*>(state);
    if (!AAState) return;

    for(int i = 0; i < actions.size(); i++)
        std::cout << "Passed course " << i <<": " << (AAState->isIthCoursePassed(i)? "Yes":"No") << std::endl;

    for(int i = 0; i < actions.size(); i++)
        std::cout << "Taken course " << i <<": " << (AAState->isIthCourseTaken(i)? "Yes":"No") << std::endl;

    //print missing courses
    std::cout << "Missing courses: ";
    for(auto req: AAState->missing_reqs)
        std::cout << req << " ";
    std::cout << std::endl;

    //print rerequites
    for (int i = 0; i < prereqs.size(); ++i) {
        std::cout << i << "<- ";
        for (int j = 0; j < prereqs[i].size(); ++j) {
            std::cout << prereqs[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

ABS::Gamestate* Model::getInitialState(int num) {
    auto* state = new AA::Gamestate();
    state->passed_courses = 0;
    state->taken_courses = 0;
    state->num_passed = 0;
    state->num_taken = 0;
    for(auto req: req_courses)
        state->missing_reqs.insert(req);
    return state;
}

ABS::Gamestate* Model::getInitialState(std::mt19937& rng)
{
    return getInitialState(0);
}

int Model::getNumPlayers() {
    return 1;
}


ABS::Gamestate* Model::copyState(ABS::Gamestate* uncasted_state) {
    auto state = dynamic_cast<Gamestate*>(uncasted_state);
    auto new_state = new Gamestate();
    *new_state = *state; //default copy constructor should work
    return new_state;
}

std::vector<int> Model::getActions_(ABS::Gamestate* uncasted_state)  {
    return actions;
}

std::pair<std::vector<double>,std::pair<int,double>> Model::applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng) {
    auto* state = dynamic_cast<AA::Gamestate*>(uncasted_state);

    float p = 1;
    double reward = 0;
    int successor = 0;

    //Reward handling and update taken course
    if(!state->isIthCourseTaken(action)) {
        state->setIthCourseTaken(action);
        state->num_taken++;
        reward -= COURSE_COST;
    }else {
        reward -= REDO_COST;
    }

    if(dense_rewards) {
        reward += state->num_passed / (double) actions.size() * PASS_REWARD;
        reward -= (state->missing_reqs.size() / (double) req_courses.size()) * INCOMPLETE_COST;
    }else {
        reward -= state->missing_reqs.empty()? 0 : INCOMPLETE_COST;
    }

    if(!state->isIthCoursePassed(action)) {
        int n_prereqs = prereqs[action].size();
        double sample = std::uniform_real_distribution<double>(0, 1)(rng);
        //No prerequisites
        if(n_prereqs == 0) {
            if(sample < PRIOR_PROB_PASS_NO_PREREQ) {
                state->setIthCoursePassed(action);
                state->num_passed++;
                state->missing_reqs.erase(action);
                p *= PRIOR_PROB_PASS_NO_PREREQ;
            }else {
                p *= 1 - PRIOR_PROB_PASS_NO_PREREQ;
                successor = 1;
            }
        //Atleast one reprequisite
        }else {
            int n_passed_prereqs = 0;
            for(auto prereq: prereqs[action]) {
                if(state->isIthCoursePassed(prereq))
                    n_passed_prereqs++;
            }
            float p_pass = PRIOR_PROB_PASS + (1 - PRIOR_PROB_PASS) * n_passed_prereqs / (1+ n_prereqs);
            if(sample < p_pass) {
                state->setIthCoursePassed(action);
                state->num_passed++;
                state->missing_reqs.erase(action);
                p *= p_pass;

            }else {
                p *= 1 - p_pass;
                successor = 1;
            }
        }
    }

    //Terminal handling
    state->terminal = state->passed_courses == (1 << actions.size()) - 1;

    return {{(double)reward}, {successor, p}};
}