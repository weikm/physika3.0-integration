/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-04-10
 * @description: test cases for Physika::World class
 * @version    : 1.0
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "framework/solver.hpp"

using namespace Physika;
using ::testing::Return;

namespace PhysikaTest {

class WorldTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        World::instance().clear();
    }
    void TearDown() override
    {
        World::instance().clear();
    }
    struct NoArgComp
    {
        NoArgComp() {}
        void reset() {}
        int  m_value = 0;
    };
    struct ArgComp
    {
        ArgComp(int value)
            : m_value(value) {}
        void reset()
        {
            m_value = 0;
        }
        int m_value = 0;
    };
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

TEST_F(WorldTest, createScene)
{
    auto scene = World::instance().createScene();
    EXPECT_TRUE(scene != nullptr);
}

TEST_F(WorldTest, destroyScene)
{
    auto scene = World::instance().createScene();
    EXPECT_TRUE(scene != nullptr);
    auto ret = World::instance().destroyScene(scene);
    EXPECT_TRUE(ret);
    ret = World::instance().destroyScene(nullptr);
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, sceneNumAndDestroyAllScenes)
{
    auto scene_a = World::instance().createScene();
    EXPECT_TRUE(scene_a != nullptr);
    auto scene_b = World::instance().createScene();
    EXPECT_TRUE(scene_b != nullptr);
    auto num = World::instance().sceneNum();
    EXPECT_EQ(num, 2);
    World::instance().destroyAllScenes();
    num = World::instance().sceneNum();
    EXPECT_EQ(num, 0);
}

TEST_F(WorldTest, getSceneById)
{
    auto scene = World::instance().createScene();
    EXPECT_TRUE(scene != nullptr);
    auto get_scene = World::instance().getSceneById(scene->id());
    EXPECT_EQ(scene, get_scene);
}

TEST_F(WorldTest, createObject)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
}

TEST_F(WorldTest, destroyObject)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto ret = World::instance().destroyObject(object);
    EXPECT_TRUE(ret);
    ret = World::instance().destroyObject(nullptr);
    EXPECT_FALSE(ret);
    // test if destroyed object will be removed from host scene
    auto scene = World::instance().createScene();
    object     = World::instance().createObject();
    scene->addObject(object);
    auto num = scene->objectNum();
    EXPECT_EQ(num, 1);
    ret = World::instance().destroyObject(object);
    num = scene->objectNum();
    EXPECT_EQ(num, 0);
}

TEST_F(WorldTest, objectNumAndDestroyAllObjects)
{
    auto obj_a = World::instance().createObject();
    EXPECT_TRUE(obj_a != nullptr);
    auto obj_b = World::instance().createObject();
    EXPECT_TRUE(obj_b != nullptr);
    auto num = World::instance().objectNum();
    EXPECT_EQ(num, 2);
    auto scene = World::instance().createScene();
    scene->addObject(obj_a);
    scene->addObject(obj_b);
    num = scene->objectNum();
    EXPECT_EQ(num, 2);
    World::instance().destroyAllObjects();
    num = World::instance().objectNum();
    EXPECT_EQ(num, 0);
    num = scene->objectNum();
    EXPECT_EQ(num, 0);
}

TEST_F(WorldTest, resetObject)
{
    auto obj_a = World::instance().createObject();
    EXPECT_TRUE(obj_a != nullptr);
    auto obj_b = World::instance().createObject();
    EXPECT_TRUE(obj_b != nullptr);
    auto ret = World::instance().resetObject(obj_a);
    EXPECT_TRUE(ret);
    ret = World::instance().resetObject(obj_b->id());
    EXPECT_TRUE(ret);
    ret = World::instance().resetObject(nullptr);
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, resetAllObjects)
{
    auto obj_a = World::instance().createObject();
    EXPECT_TRUE(obj_a != nullptr);
    auto obj_b = World::instance().createObject();
    EXPECT_TRUE(obj_b != nullptr);
    auto ret = World::instance().resetAllObjects();
    EXPECT_TRUE(ret);
}

TEST_F(WorldTest, getObjectById)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto get_object = World::instance().getObjectById(object->id());
    EXPECT_EQ(object, get_object);
}

TEST_F(WorldTest, getObjectId)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto obj_id       = object->id();
    auto world_obj_id = World::instance().getObjectId(object);
    EXPECT_EQ(obj_id, world_obj_id);
}

TEST_F(WorldTest, addComponentToObject)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto ret = World::instance().addComponentToObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().addComponentToObject<ArgComp>(object, 1);
    EXPECT_TRUE(ret);
    ret = World::instance().addComponentToObject<NoArgComp>(nullptr);
    EXPECT_FALSE(ret);
    ret = World::instance().addComponentToObject<ArgComp>(nullptr, 1);
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, addComponentToObjectById)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto ret = World::instance().addComponentToObject<NoArgComp>(object->id());
    EXPECT_TRUE(ret);
    ret = World::instance().addComponentToObject<ArgComp>(object->id(), 1);
    EXPECT_TRUE(ret);
    ret = World::instance().addComponentToObject<NoArgComp>(10000);  // id that does not exist
    EXPECT_FALSE(ret);
    ret = World::instance().addComponentToObject<ArgComp>(10000, 1);
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, checkObjectHasComponent)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto ret = World::instance().addComponentToObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().checkObjectHasComponent<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().checkObjectHasComponent<ArgComp>(object);
    EXPECT_FALSE(ret);
    ret = World::instance().checkObjectHasComponent<NoArgComp>(object->id());
    EXPECT_TRUE(ret);
    ret = World::instance().checkObjectHasComponent<ArgComp>(object->id());
    EXPECT_FALSE(ret);
    ret = World::instance().checkObjectHasComponent<NoArgComp>(nullptr);
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, removeComponentFromObject)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto ret = World::instance().addComponentToObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().checkObjectHasComponent<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().removeComponentFromObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().checkObjectHasComponent<ArgComp>(object);
    EXPECT_FALSE(ret);
    ret = World::instance().removeComponentFromObject<NoArgComp>(object);  // remove component which the object doesn't have
    EXPECT_FALSE(ret);
    ret = World::instance().addComponentToObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().checkObjectHasComponent<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().removeComponentFromObject<NoArgComp>(object->id());
    EXPECT_TRUE(ret);
    ret = World::instance().checkObjectHasComponent<ArgComp>(object);
    EXPECT_FALSE(ret);
    ret = World::instance().removeComponentFromObject<NoArgComp>(object->id());  // remove component which the object doesn't have
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, removeAllComponentsFromObject)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto ret = World::instance().addComponentToObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().addComponentToObject<ArgComp>(object, 1);
    EXPECT_TRUE(ret);
    ret = World::instance().removeAllComponentsFromObject(object);
    EXPECT_TRUE(ret);
    ret = World::instance().removeAllComponentsFromObject(object);
    EXPECT_TRUE(ret);
    ret = World::instance().removeAllComponentsFromObject(nullptr);
    EXPECT_FALSE(ret);
    ret = World::instance().addComponentToObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    ret = World::instance().addComponentToObject<ArgComp>(object, 1);
    EXPECT_TRUE(ret);
    ret = World::instance().removeAllComponentsFromObject(object->id());
    EXPECT_TRUE(ret);
    ret = World::instance().removeAllComponentsFromObject(object->id());
    EXPECT_TRUE(ret);
    ret = World::instance().removeAllComponentsFromObject(10000);  // object id that does not exist
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, objectComponentNumAndGetComponent)
{
    auto object = World::instance().createObject();
    EXPECT_TRUE(object != nullptr);
    auto ret = World::instance().addComponentToObject<NoArgComp>(object);
    EXPECT_TRUE(ret);
    auto num = World::instance().objectComponentNum(object);
    EXPECT_EQ(num, 1);
    num = World::instance().objectComponentNum(nullptr);
    EXPECT_EQ(num, 0);
    ret = World::instance().addComponentToObject<ArgComp>(object, 1);
    EXPECT_TRUE(ret);
    num = World::instance().objectComponentNum(object->id());
    EXPECT_EQ(num, 2);
    num = World::instance().objectComponentNum(10000);  // object id that does not exist
    EXPECT_EQ(num, 0);
    auto com_by_obj    = World::instance().getObjectComponent<NoArgComp>(object);
    auto com_by_obj_id = World::instance().getObjectComponent<NoArgComp>(object->id());
    EXPECT_EQ(com_by_obj, com_by_obj_id);
    EXPECT_TRUE(com_by_obj != nullptr);
}

TEST_F(WorldTest, createAndDestroySolver)
{
    auto solver = World::instance().createSolver<MockSolver>();
    EXPECT_TRUE(solver != nullptr);
    auto ret = World::instance().destroySolver(solver);
    EXPECT_TRUE(ret);
    ret = World::instance().destroySolver(nullptr);
    EXPECT_FALSE(ret);
    // test if destroyed solver will be removed from host scene
    solver     = World::instance().createSolver<MockSolver>();
    auto scene = World::instance().createScene();
    scene->addSolver(solver);
    ret      = World::instance().destroySolver(solver);
    auto num = scene->solverNum();
    EXPECT_EQ(num, 0);
    // test destroy all solvers
    solver        = World::instance().createSolver<MockSolver>();
    auto solver_2 = World::instance().createSolver<MockSolver>();
    scene->addSolver(solver);
    scene->addSolver(solver_2);
    num = scene->solverNum();
    EXPECT_EQ(num, 2);
    World::instance().destroyAllSolvers();
    num = scene->solverNum();
    EXPECT_EQ(num, 0);
}

TEST_F(WorldTest, getSolverById)
{
    MockSolver dangling_solver;
    auto       solver = World::instance().getSolverById(dangling_solver.id());
    EXPECT_TRUE(solver == nullptr);
    auto registered_solver = World::instance().createSolver<MockSolver>();
    auto get_solver        = World::instance().getSolverById(registered_solver->id());
    EXPECT_EQ(registered_solver, get_solver);
    // test isDanglingSolver
    auto ret = World::instance().isDanglingSolver(&dangling_solver);
    EXPECT_TRUE(ret);
    ret = World::instance().isDanglingSolver(registered_solver);
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, clear)
{
    auto scene  = World::instance().createScene();
    auto obj    = World::instance().createObject();
    auto solver = World::instance().createSolver<MockSolver>();
    scene->addObject(obj);
    scene->addSolver(solver);
    auto num = World::instance().sceneNum();
    EXPECT_EQ(num, 1);
    num = World::instance().objectNum();
    EXPECT_EQ(num, 1);
    num = World::instance().solverNum();
    EXPECT_EQ(num, 1);
    World::instance().clear();
    num = World::instance().sceneNum();
    EXPECT_EQ(num, 0);
    num = World::instance().objectNum();
    EXPECT_EQ(num, 0);
    num = World::instance().solverNum();
    EXPECT_EQ(num, 0);
}

TEST_F(WorldTest, reset)
{
    auto scene  = World::instance().createScene();
    auto obj    = World::instance().createObject();
    auto solver = World::instance().createSolver<MockSolver>();
    scene->addObject(obj);
    scene->addSolver(solver);
    EXPECT_CALL(*solver, reset).WillRepeatedly(Return(true));
    auto num = World::instance().sceneNum();
    EXPECT_EQ(num, 1);
    num = World::instance().objectNum();
    EXPECT_EQ(num, 1);
    num = World::instance().solverNum();
    EXPECT_EQ(num, 1);
    auto ret = World::instance().reset();
    EXPECT_TRUE(ret);
    num = World::instance().sceneNum();
    EXPECT_EQ(num, 1);
    num = World::instance().objectNum();
    EXPECT_EQ(num, 1);
    num = World::instance().solverNum();
    EXPECT_EQ(num, 1);
}

TEST_F(WorldTest, step)
{
    auto good_scene  = World::instance().createScene();
    auto good_solver = World::instance().createSolver<MockSolver>();
    good_scene->addSolver(good_solver);
    EXPECT_CALL(*good_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(*good_solver, initialize).WillRepeatedly(Return(true));
    EXPECT_CALL(*good_solver, step).WillRepeatedly(Return(true));
    EXPECT_CALL(*good_solver, clearAttachment).WillRepeatedly(Return());
    auto ret = World::instance().step();
    EXPECT_TRUE(ret);
    auto bad_scene  = World::instance().createScene();
    auto bad_solver = World::instance().createSolver<MockSolver>();
    bad_scene->addSolver(bad_solver);
    EXPECT_CALL(*bad_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(*bad_solver, initialize).WillRepeatedly(Return(false));
    EXPECT_CALL(*bad_solver, step).WillRepeatedly(Return(false));
    EXPECT_CALL(*bad_solver, clearAttachment).WillRepeatedly(Return());
    ret = World::instance().step();
    EXPECT_FALSE(ret);
}

TEST_F(WorldTest, run)
{
    auto good_scene  = World::instance().createScene();
    auto good_solver = World::instance().createSolver<MockSolver>();
    good_scene->addSolver(good_solver);
    EXPECT_CALL(*good_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(*good_solver, initialize).WillRepeatedly(Return(true));
    EXPECT_CALL(*good_solver, run).WillRepeatedly(Return(true));
    EXPECT_CALL(*good_solver, clearAttachment).WillRepeatedly(Return());
    auto ret = World::instance().run();
    EXPECT_TRUE(ret);
    auto bad_scene  = World::instance().createScene();
    auto bad_solver = World::instance().createSolver<MockSolver>();
    bad_scene->addSolver(bad_solver);
    EXPECT_CALL(*bad_solver, isInitialized).WillOnce(Return(false)).WillRepeatedly(Return(true));
    EXPECT_CALL(*bad_solver, initialize).WillRepeatedly(Return(false));
    EXPECT_CALL(*bad_solver, run).WillRepeatedly(Return(false));
    EXPECT_CALL(*bad_solver, clearAttachment).WillRepeatedly(Return());
    ret = World::instance().run();
    EXPECT_FALSE(ret);
}

}  // namespace PhysikaTest