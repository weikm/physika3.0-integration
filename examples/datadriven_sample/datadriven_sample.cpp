/**
 * @author     : Zirui Dong (2213346189@qq.com)
 * @date       : 2023-3-20
 * @description: A sample of using Physika
 * @version    : 1.0
 */

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include "/datadriven/datadriven_solver.h"
using namespace Physika;

int main() {
    try {
        // 初始化 DataDrivenSolver，传入必要的参数
        Physika::DataDrivenSolver solver("datadriven_solver", 
                                "Worker",
                                std::string(SOURCE_PATH) + "/pretrained_model_weights.pt", 
                                std::string(SOURCE_PATH) + "/example_scene.json", 
                                std::string(SOURCE_PATH) + "/example_out",
                                500, 1);
        import_array();
        // 开始测试运行速度指标
        std::cout << "Executing steps..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        solver.steps();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);
        std::cout << "Steps execution completed." << std::endl;
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
        std::cout << "fps: " << 500 / duration.count() << std::endl;
        std::string datasetDir = std::string(SOURCE_PATH) + "/datasets/";
        solver.evaluate(datasetDir);
        
        // 重新执行一次仿真，用于保存数据
        solver.writePly = true;
        PyObject* pGetFluidsResult = solver.getFluids();
        if (!pGetFluidsResult || !PyList_Check(pGetFluidsResult) || PyList_Size(pGetFluidsResult) < 1) {
            std::cerr << "Failed to get fluids data or data format is incorrect." << std::endl;
            Py_XDECREF(pGetFluidsResult);
            return -1;
        }
        PyObject* pfluids = PyList_GetItem(pGetFluidsResult, 0);
        if (!PyList_Check(pfluids) || PyList_Size(pfluids) < 2) {
            std::cerr << "Fluid data format is incorrect." << std::endl;
            Py_DECREF(pfluids);
            return -1;
        }
        PyObject* pPos = PyList_GetItem(pfluids, 0);
        PyObject* pVel = PyList_GetItem(pfluids, 1);
        if (PyArray_Check(pPos) && PyArray_Check(pVel)) {
            for(int i = 0; i < solver.numIterations; i++) {
                PyObject* pStepOneResult = solver.stepOne(pPos, pVel, i);
                pPos = PyList_GetItem(pStepOneResult, 0);
                pVel = PyList_GetItem(pStepOneResult, 1);
                if (i % 10 ==0) {
                    std::cout << "step " << i << " ";
                    solver.printNpyShape(pPos, "simulation particles shape");
                }
            }
        }    
        

        // 清理
        Py_DECREF(pGetFluidsResult);

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}

