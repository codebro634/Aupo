cmake_minimum_required(VERSION 3.16.3)
project(Aupo)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(Aupo src/main.cpp
        src/Arena.cpp
        src/Agents/Mcts/MctsAgent.cpp
        src/Agents/Mcts/MctsNode.cpp
        include/Games/Gamestate.h
        include/Agents/Agent.h
        include/Agents/Mcts/MctsAgent.h
        include/Agents/Mcts/MctsNode.h
        include/Arena.h
        include/Games/MDPs/GameOfLife.h
        src/Games/MDPs/GameOfLife.cpp
        include/Utils/Debug.h
        include/Utils/UnitTest.h
        include/Games/MDPs/SysAdmin.h
        src/Games/MDPs/SysAdmin.cpp
        include/Games/MDPs/AcademicAdvising.h
        src/Games/MDPs/AcademicAdvising.cpp
        include/Utils/Argparse.h
        include/Games/MDPs/Tamarisk.h
        src/Games/MDPs/Tamarisk.cpp
        include/Agents/Aupo/AupoNode.h
        include/Agents/Aupo/AupoAgent.h
        src/Agents/Aupo/AupoAgent.cpp
        src/Agents/Aupo/AupoNode.cpp
        include/Utils/Distributions.h
        src/Utils/Distributions.cpp
        include/Games/MDPs/Wildfire.h
        src/Games/MDPs/Wildfire.cpp
        include/Games/MDPs/MultiArmedBandit.h
        src/Games/MDPs/MultiArmedBandit.cpp
)
