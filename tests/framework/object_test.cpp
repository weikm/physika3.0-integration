/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-04-07
 * @description: test cases for Physika::Object class, Scene related API is tested in scene test
 * @version    : 1.0
 */

#include "gtest/gtest.h"

#include "framework/object.hpp"
#include "framework/world.hpp"

using namespace Physika;
namespace PhysikaTest {

class ObjectTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        World::instance().clear();
        m_object = World::instance().createObject();
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

    Object* m_object;
};

TEST_F(ObjectTest, addComponent)
{
    EXPECT_TRUE(m_object != nullptr);
    auto ret = m_object->addComponent<NoArgComp>();
    EXPECT_TRUE(ret);
    ret = m_object->addComponent<NoArgComp>();  // redundant component type
    EXPECT_FALSE(ret);
    ret = m_object->addComponent<ArgComp>(1);
    EXPECT_TRUE(ret);
}

TEST_F(ObjectTest, hasComponent)
{
    EXPECT_TRUE(m_object != nullptr);
    auto ret = m_object->hasComponent<int>();
    EXPECT_FALSE(ret);
    ret = m_object->addComponent<NoArgComp>();
    EXPECT_TRUE(ret);
    ret = m_object->hasComponent<NoArgComp>();
    EXPECT_TRUE(ret);
}

TEST_F(ObjectTest, removeComponent)
{
    EXPECT_TRUE(m_object != nullptr);
    auto ret = m_object->removeComponent<NoArgComp>();  // component does not exist
    EXPECT_FALSE(ret);
    ret = m_object->addComponent<NoArgComp>();
    EXPECT_TRUE(ret);
    ret = m_object->removeComponent<NoArgComp>();
    EXPECT_TRUE(ret);
    ret = m_object->addComponent<ArgComp>(1);
    EXPECT_TRUE(ret);
    ret = m_object->removeComponent<ArgComp>();
    EXPECT_TRUE(ret);
}

TEST_F(ObjectTest, componentNum)
{
    EXPECT_TRUE(m_object != nullptr);
    auto num = m_object->componentNum();
    EXPECT_EQ(num, 0);
    auto ret = m_object->addComponent<NoArgComp>();
    EXPECT_TRUE(ret);
    num = m_object->componentNum();
    EXPECT_EQ(num, 1);
    ret = m_object->addComponent<ArgComp>(1);
    EXPECT_TRUE(ret);
    num = m_object->componentNum();
    EXPECT_EQ(num, 2);
    ret = m_object->removeComponent<ArgComp>();
    EXPECT_TRUE(ret);
    num = m_object->componentNum();
    EXPECT_EQ(num, 1);
    ret = m_object->removeAllComponents();
    EXPECT_TRUE(ret);
    num = m_object->componentNum();
    EXPECT_EQ(num, 0);
}

TEST_F(ObjectTest, getComponent)
{
    EXPECT_TRUE(m_object != nullptr);
    auto* null_comp = m_object->getComponent<ArgComp>();
    EXPECT_TRUE(null_comp == nullptr);
    auto ret = m_object->addComponent<ArgComp>(1);
    EXPECT_TRUE(ret);
    auto* comp = m_object->getComponent<ArgComp>();
    EXPECT_TRUE(comp != nullptr);
}

TEST_F(ObjectTest, reset)
{
    EXPECT_TRUE(m_object != nullptr);
    auto ret = m_object->addComponent<ArgComp>(1);
    EXPECT_TRUE(ret);
    ret = m_object->reset();
    EXPECT_TRUE(ret);
    auto* comp = m_object->getComponent<ArgComp>();
    EXPECT_EQ(comp->m_value, 0);
}

}  // namespace PhysikaTest