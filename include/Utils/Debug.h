//
// Created by chris on 11.10.2024.
//

#ifndef DEBUG_H
#define DEBUG_H

#include<fstream>

#endif //DEBUG_H

#define DEBUG_

#ifdef DEBUG
    #include <iostream>
    #define DEBUG_LOG(...) std::cout << __VA_ARGS__ << std::endl;
    #define DEBUG_CALL(function) function;
    #define DEBUG_WAIT() std::cin.get();
#else
    #define DEBUG_LOG(...)
    #define DEBUG_CALL(function)
    #define DEBUG_WAIT()
#endif

#define VISUALIZE_

#ifdef VISUALIZE
    #include <iostream>
    #include <thread>
    #include <chrono>
    #define VISUALIZE_CALL(function) function;
    #define VISUALIZE_WAIT() std::cin.get();
    // #define VISUALIZE_WAIT() std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#define VISUALIZE_DUMP_TREE(tree) std::ofstream file("tree.txt", std::ios::out); if (!file.is_open()) std::cerr << "Failed to open file for writing tree" << std::endl; tree->dumpTree(file);  file.close(); std::cout << "Tree written to file" << std::endl;
#else
    #define VISUALIZE_CALL(function)
    #define VISUALIZE_WAIT()
    #define VISUALIZE_DUMP_TREE(tree)
#endif
