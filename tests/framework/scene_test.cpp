/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-04-09
 * @description: test cases for Physika::Scene class
 * @version    : 1.0
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "framework/object.hpp"
#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/solver.hpp"

using namespace Physika;
using ::testing::Return;

namespace PhysikaTest {

class SceneTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        World::instance().clear();
        for (int i = 0; i < 2; ++i)
        {
            m_scene[i]  = World::instance().createScene();
            m_object[i] = World::instance().createObject();
        }
    }
    void TearDown() override
    {
        World::instance().clear();
    }

    Scene*  m_scene[2];
    Object* m_object[2];
};

class MockSolver : public Solver
{
public:
    MOCK_METHOD(bool, initialize, (), (override));
    MOCK_METHOD(bool, isInitialized, (), (const, override));
    MOCK_METHOD(bool, reset, (), (override));
    MOCK_METHOD(bool, step, (), (override));
    MOCK_METHOD(bool, run, (), (override));
    MOCK_METHOD(bool, isApplicable, (const Object* object), (const, override));
    MOCK_METHOD(bool, attachObject, (Object * object), (override));
    MOCK_METHOD(bool, detachObject, (Object * object), (override));
    MOCK_METHOD(void, clearAttachment, (), (override));
};

TEST_F(SceneTest, addObject)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    EXPECT_TRUE(m_object[0] != nullptr);
    EXPECT_TRUE(m_scene[1] != nullptr);
    EXPECT_TRUE(m_object[1] != nullptr);
    auto ret = m_scene[0]->addObject(m_object[0]);
    EXPECT_TRUE(ret);
    ret = m_scene[0]->addObject(nullptr);  // add nullptr
    EXPECT_FALSE(ret);
    ret = m_scene[1]->addObject(m_object[1]);
    EXPECT_TRUE(ret);
    ret = m_scene[0]->addObject(m_object[1]);  // object in another scene
    EXPECT_FALSE(ret);
}

TEST_F(SceneTest, addObjectById)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    EXPECT_TRUE(m_object[0] != nullptr);
    EXPECT_TRUE(m_scene[1] != nullptr);
    EXPECT_TRUE(m_object[1] != nullptr);
    auto ret = m_scene[0]->addObjectById(m_object[0]->id());
    EXPECT_TRUE(ret);
    ret = m_scene[0]->addObjectById(10000);  // add invalid object
    EXPECT_FALSE(ret);
    ret = m_scene[1]->addObjectById(m_object[1]->id());
    EXPECT_TRUE(ret);
    ret = m_scene[0]->addObjectById(m_object[1]->id());  // object in another scene
    EXPECT_FALSE(ret);
}

TEST_F(SceneTest, removeObject)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    EXPECT_TRUE(m_object[0] != nullptr);
    auto ret = m_scene[0]->removeObject(nullptr);
    EXPECT_FALSE(ret);
    ret = m_scene[0]->removeObject(m_object[0]);  // remove object not in scene
    EXPECT_FALSE(ret);
    ret = m_scene[0]->addObject(m_object[0]);
    EXPECT_TRUE(ret);
    ret = m_scene[0]->removeObject(m_object[0]);
    EXPECT_TRUE(ret);
}

TEST_F(SceneTest, removeObjectById)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    EXPECT_TRUE(m_object[0] != nullptr);
    auto ret = m_scene[0]->removeObjectById(10000);  // remove object not exist
    EXPECT_FALSE(ret);
    ret = m_scene[0]->removeObjectById(m_object[0]->id());  // remove object not in scene
    EXPECT_FALSE(ret);
    ret = m_scene[0]->addObjectById(m_object[0]->id());
    EXPECT_TRUE(ret);
    ret = m_scene[0]->removeObjectById(m_object[0]->id());
    EXPECT_TRUE(ret);
}

TEST_F(SceneTest, objectNumAndRemoveAllObjects)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    EXPECT_TRUE(m_object[0] != nullptr);
    auto ret = m_scene[0]->addObject(m_object[0]);
    EXPECT_TRUE(ret);
    auto num = m_scene[0]->objectNum();
    EXPECT_EQ(num, 1);
    ret = m_scene[0]->addObject(m_object[1]);
    EXPECT_TRUE(ret);
    num = m_scene[0]->objectNum();
    EXPECT_EQ(num, 2);
    m_scene[0]->removeAllObjects();
    num = m_scene[0]->objectNum();
    EXPECT_EQ(num, 0);
}

TEST_F(SceneTest, getObjectById)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    EXPECT_TRUE(m_object[0] != nullptr);
    auto ret = m_scene[0]->addObject(m_object[0]);
    EXPECT_TRUE(ret);
    auto* obj = m_scene[0]->getObjectById(m_object[0]->id());
    EXPECT_EQ(obj, m_object[0]);
    obj = m_scene[0]->getObjectById(m_object[1]->id());
    EXPECT_EQ(obj, nullptr);
}

TEST_F(SceneTest, addSolver)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_solver;
    auto       ret = m_scene[0]->addSolver(&mock_solver);  // dangling solver
    EXPECT_TRUE(ret);
    auto* registered_mock_solver = World::instance().createSolver<MockSolver>();
    ret                          = m_scene[0]->addSolverById(registered_mock_solver->id());  // registered solver
    EXPECT_TRUE(ret);
    ret = m_scene[0]->addSolver(nullptr);
    EXPECT_FALSE(ret);
}

TEST_F(SceneTest, addSolverById)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_solver;
    auto       ret = m_scene[0]->addSolverById(mock_solver.id());  // dangling solver
    EXPECT_FALSE(ret);
    auto* registered_mock_solver = World::instance().createSolver<MockSolver>();
    ret                          = m_scene[0]->addSolverById(registered_mock_solver->id());  // registered solver
    EXPECT_TRUE(ret);
}

TEST_F(SceneTest, removeSolver)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_solver;
    auto       ret = m_scene[0]->addSolver(&mock_solver);  // dangling solver
    EXPECT_TRUE(ret);
    auto* registered_mock_solver = World::instance().createSolver<MockSolver>();
    ret                          = m_scene[0]->addSolverById(registered_mock_solver->id());  // registered solver
    EXPECT_TRUE(ret);
    auto rm_ret = m_scene[0]->removeSolver(&mock_solver);
    EXPECT_TRUE(rm_ret);
    rm_ret = m_scene[0]->removeSolver(registered_mock_solver);
    EXPECT_TRUE(rm_ret);
    rm_ret = m_scene[0]->removeSolver(registered_mock_solver);  // solver not in scene
    EXPECT_FALSE(rm_ret);
    rm_ret = m_scene[0]->removeSolver(nullptr);
    EXPECT_FALSE(rm_ret);
}

TEST_F(SceneTest, removeSolverById)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_solver;
    auto       ret = m_scene[0]->addSolver(&mock_solver);  // dangling solver
    EXPECT_TRUE(ret);
    auto* registered_mock_solver = World::instance().createSolver<MockSolver>();
    ret                          = m_scene[0]->addSolverById(registered_mock_solver->id());  // registered solver
    EXPECT_TRUE(ret);
    auto rm_ret = m_scene[0]->removeSolverById(mock_solver.id());
    EXPECT_TRUE(rm_ret);
    rm_ret = m_scene[0]->removeSolverById(registered_mock_solver->id());
    EXPECT_TRUE(rm_ret);
    rm_ret = m_scene[0]->removeSolverById(registered_mock_solver->id());  // solver not in scene
    EXPECT_FALSE(rm_ret);
}

TEST_F(SceneTest, solverNumAndRemoveAllSolvers)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_solver;
    auto       ret = m_scene[0]->addSolver(&mock_solver);  // dangling solver
    EXPECT_TRUE(ret);
    auto* registered_mock_solver = World::instance().createSolver<MockSolver>();
    ret                          = m_scene[0]->addSolverById(registered_mock_solver->id());  // registered solver
    EXPECT_TRUE(ret);
    auto num = m_scene[0]->solverNum();
    EXPECT_EQ(num, 2);
    m_scene[0]->removeAllSolvers();
    num = m_scene[0]->solverNum();
    EXPECT_EQ(num, 0);
}

TEST_F(SceneTest, getSolverById)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_solver;
    auto       solver_ret = m_scene[0]->getSolverById(mock_solver.id());
    EXPECT_TRUE(solver_ret == nullptr);
    auto ret = m_scene[0]->addSolver(&mock_solver);  // dangling solver
    EXPECT_TRUE(ret);
    solver_ret = m_scene[0]->getSolverById(mock_solver.id());
    EXPECT_TRUE(solver_ret != nullptr);
}

TEST_F(SceneTest, step)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_good_solver;
    EXPECT_CALL(mock_good_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_good_solver, initialize).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_good_solver, step).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_good_solver, clearAttachment).WillRepeatedly(Return());
    MockSolver mock_bad_solver;
    EXPECT_CALL(mock_bad_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_bad_solver, initialize).WillRepeatedly(Return(false));
    EXPECT_CALL(mock_bad_solver, step).WillRepeatedly(Return(false));
    EXPECT_CALL(mock_bad_solver, clearAttachment).WillRepeatedly(Return());
    auto ret = m_scene[0]->addSolver(&mock_good_solver);
    EXPECT_TRUE(ret);
    ret = m_scene[0]->step();
    EXPECT_TRUE(ret);
    ret = m_scene[0]->addSolver(&mock_bad_solver);
    EXPECT_TRUE(ret);
    ret = m_scene[0]->step();
    EXPECT_FALSE(ret);
}

TEST_F(SceneTest, run)
{
    EXPECT_TRUE(m_scene[0] != nullptr);
    MockSolver mock_good_solver;
    EXPECT_CALL(mock_good_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_good_solver, initialize).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_good_solver, run).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_good_solver, clearAttachment).WillRepeatedly(Return());
    MockSolver mock_bad_solver;
    EXPECT_CALL(mock_bad_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(mock_bad_solver, initialize).WillRepeatedly(Return(false));
    EXPECT_CALL(mock_bad_solver, run).WillRepeatedly(Return(false));
    EXPECT_CALL(mock_bad_solver, clearAttachment).WillRepeatedly(Return());
    auto ret = m_scene[0]->addSolver(&mock_good_solver);
    EXPECT_TRUE(ret);
    ret = m_scene[0]->run();
    EXPECT_TRUE(ret);
    ret = m_scene[0]->addSolver(&mock_bad_solver);
    EXPECT_TRUE(ret);
    ret = m_scene[0]->run();
    EXPECT_FALSE(ret);
}

}  // namespace PhysikaTest