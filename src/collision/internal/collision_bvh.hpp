#pragma once
#include "collision/interface/collision_data.hpp"
#include "collision/internal/collision_box.hpp"
#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/trimesh.hpp"

#include <vector>

namespace Physika {
class triMesh;
class qbvh;
class front_node;
class bvh_node;

static vec3f* s_fcenters;  //!< used to store temporary data

    /**
 * front list of the front node
 */
class front_list : public std::vector<front_node>
{
public:
    /**
     * @brief push the data structure to gpu
     *
     * @param[in] r1 bvh node 1
     * @param[in] r2 bvh node 2
     */
    void push2GPU(bvh_node* r1, bvh_node* r2 = NULL);
};

/**
 * bvh node
 */
class bvh_node
{
public:
    BOX<REAL>        _box;       //!< aabb box
    static bvh_node* s_current;  //!< store temporary data
    int              _child;     //!< child >=0 leaf with tri_id, <0 left & right
    int              _parent;    //!< parent

    void setParent(int p);

public:
    /**
     * @brief constructor
     */
    bvh_node();

    /**
     * @brief destructor
     */
    ~bvh_node();

    /**
     * @brief check collision with the other bvh node
     *
     * @param[in]  other another bvh node
     * @param[out] ret   triangle pair to store the result
     */
    void collide(bvh_node* other, std::vector<TrianglePair>& ret);

    void collide(bvh_node* other, front_list& f, int level, int ptr);

    /**
     * @brief construct the bvh node
     *
     * @param[in] id        index
     * @param[in] s_fboxes  boxes
     */
    void construct(unsigned int id, BOX<REAL>* s_fboxes);

    /**
     * @brief construct the bvh node
     *
     * @param[in] lst        list
     * @param[in] num        list item number
     * @param[in] s_fcenters centers
     * @param[in] s_fboxes   boxes
     * @param[in] s_current  current node
     */
    void construct(
        unsigned int* lst,
        unsigned int  num,
        vec3f*        s_fcenters,
        BOX<REAL>*    s_fboxes,
        bvh_node*&    s_current);

    /**
     * @brief refit algorithm, internal
     *
     * @param[in] s_fboxes boxes
     */
    void refit(BOX<REAL>* s_fboxes);

    /**
     * @brief reset parents of the tree, internal
     *
     * @param[in] root       tree node
     */
    void resetParents(bvh_node* root);

    /**
     * @brief get boxes
     *
     * @return boxes
     */
    FORCEINLINE BOX<REAL>& box()
    {
        return _box;
    }

    /**
     * @brief get left child
     *
     * @return left child
     */
    FORCEINLINE bvh_node* left()
    {
        return this - _child;
    }

    /**
     * @brief get right child
     *
     * @return right child
     */
    FORCEINLINE bvh_node* right()
    {
        return this - _child + 1;
    }

    /**
     * @brief get triangle id
     *
     * @return triangle id
     */
    FORCEINLINE int triID()
    {
        return _child;
    }

    /**
     * @brief check leaf node
     *
     * @return whether current node is a leaf node
     */
    FORCEINLINE int isLeaf()
    {
        return _child >= 0;
    }

    /**
     * @brief get parent
     *
     * @return parent id
     */
    FORCEINLINE int parentID()
    {
        return _parent;
    }

    /**
     * @brief get current level
     *
     * @param[in]  current   current index
     * @param[out] max_level max level
     */
    FORCEINLINE void getLevel(int current, int& max_level);

    FORCEINLINE void self_collide(front_list& lst, bvh_node* r);

    /**
     * @brief get level index
     *
     * @param[in]  current current index
     * @param[in] idx      index
     */
    FORCEINLINE void getLevelIdx(int current, unsigned int* idx);

    /**
     * @brief sprout algorithm
     *
     * @param[in]  other   another bvh node
     * @param[out] append  front list
     * @param[out] ret     result in triangle pairs
     */
    void sprouting(bvh_node* other, front_list& append, std::vector<TrianglePair>& ret);

    /**
     * @brief sprout algorithm second version
     *
     * @param[in]  other   another bvh node
     * @param[out] append  front list
     * @param[out] ret     result in triangle pairs
     */
    void sprouting2(bvh_node* other, front_list& append, std::vector<TrianglePair>& ret);

    /**
     * @brief visualize the bvh node
     *
     */
    void visualize(int level)
    {
        if (isLeaf())
        {
            _box.visualize();
        }
        else if ((level > 0))
        {
            if (level == 1)
            {
                _box.visualize();
            }
            else
            {
                if (left())
                    left()->visualize(level - 1);
                if (right())
                    right()->visualize(level - 1);
            }
        }
    }

    friend class bvh;
    friend class qbvh_node;
    friend class sbvh_node;
};

/**
 * front node in the bvtt front list
 */
class front_node
{
public:
    bvh_node*    _left;   //!< left child
    bvh_node*    _right;  //!< right child
    unsigned int _flag;   //!< vailid or not
    unsigned int _ptr;    //!< self-coliding parent;

    /**
     * @brief constructor
     *
     * @param[in] l   left child
     * @param[in] r   right child
     * @param[in] ptr parant
     */
    front_node(bvh_node* l, bvh_node* r, unsigned int ptr);
};

/**
 * internal class for bvh construction
 */
class aap
{
public:
    int   _xyz;  //!< type
    float _p;    //!< center

    /**
     * @brief constructor
     *
     * @param[in] total box in whole
     */
    FORCEINLINE aap(const BOX<REAL>& total)
    {
        vec3f center = vec3f(total.center());
        int   xyz    = 2;

        if (total.width() >= total.height() && total.width() >= total.depth())
        {
            xyz = 0;
        }
        else if (total.height() >= total.width() && total.height() >= total.depth())
        {
            xyz = 1;
        }

        _xyz = xyz;
        _p   = center[xyz];
    }

    /**
     * @brief check if center is inside
     *
     * @param[in] mid mid point
     * @return whether inside
     */
    inline bool inside(const vec3f& mid) const
    {
        return mid[_xyz] > _p;
    }
};

/**
 * bvh class for acceleration
 */
class bvh
{
    int           _num         = 0;        //!< node number
    bvh_node*     _nodes       = nullptr;  //!< node array pointer
    BOX<REAL>*    s_fboxes     = nullptr;  //!< boxes
    unsigned int* s_idx_buffer = nullptr;  //!< index buffeer
public:
    /**
     * @brief default constuctor
     */
    bvh(){};

    bvh(std::vector<triMesh*>& ms);

    bvh(triMesh* mesh);
    /**
     * @brief refit algorithm, used for bvh construction
     *
     * @param s_fboxes aabb boxes
     */
    void refit(BOX<REAL>* s_fboxes);

    void refit(std::vector<triMesh*>& ms, REAL tol);

    /**
     * @brief construct algorithm
     *
     * @param[in] ms triangle meshes
     */
    void construct(std::vector<triMesh*>& ms);

    void construct(triMesh* mesh);

    /**
     * @brief reorder algorithm, used for construction
     */
    void reorder();  // for breath-first refit

    /**
     * @brief reset parent algorithm
     */
    void resetParents();

    /**
     * @brief destructor
     */
    ~bvh()
    {
        if (_nodes)
            delete[] _nodes;
    }

    /**
     * @return the num of nodes
     */
    int num() const
    {
        return _num;
    }

    /**
     * @brief get root node
     *
     * @return root node
     */
    bvh_node* root() const
    {
        return _nodes;
    }

    /**
     * @brief refit algorithm
     *
     * @param[in] ms triangle mesh
     */
    void refit(std::vector<triMesh*>& ms);

    /**
     * @brief push the host bvh structure to device bvh structure
     *
     * @param[in] isCloth is cloth
     */
    void push2GPU(bool);

    /**
     * @brief collide with other bvh structure
     *
     * @param[in]  other another bvh structure
     * @param[out] f     bvtt front
     *
     */
    void collide(bvh* other, front_list& f);

    /**
     * @brief collide with other bvh structure
     *
     * @param[in]  other another bvh structure
     * @param[out] ret   triangle pair as result
     */
    void collide(bvh* other, std::vector<TrianglePair>& ret);

    /**
     * @brief self collision detection
     *
     * @param[out] f bvtt front
     * @param[in]  c triangle mesh
     */
    void self_collide(front_list& f, std::vector<triMesh*>& c);  // hxl



    void rayCasting(const vec3f& pt, const vec3f& dir, triMesh* km, REAL& ret);
    // faster with fewer division
    void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f& dir2, triMesh* km, REAL& ret);
    // faster with no recursion
    void rayCasting3(const vec3f& pt, const vec3f& dir, const vec3f& dir2, triMesh* km, REAL& ret);

    void visualize(int);

    friend class qbvh;
    friend class sbvh;
};

class CollisionManager
{
public:
    CollisionManager(){};
    ~CollisionManager() {}
    static std::vector<triMesh*> Meshes;  //!< storage for input mesh

    /**
     * get the id of the mesh
     *
     * @param[in]  id  id
     * @param[in]  m   mesh
     * @param[out] mid mesh id
     * @param[out] fid face id
     */
    static void mesh_id(int id, std::vector<triMesh*>& m, int& mid, int& fid)
    {
        //fid = id;
        //for (mid = 0; mid < m.size(); mid++)
        //    if (fid < m[mid]->_num_tri)
        //    {
        //        return;
        //    }
        //    else
        //    {
        //        fid -= m[mid]->_num_tri;
        //    }

        //assert(false);
        //fid = -1;
        //mid = -1;
        //printf("mesh_id error!!!!\n");
        //abort();
    }

    /**
     * check whether two mesh specified by ids is covertex
     *
     * @param[in] id1 mesh id1
     * @param[in] id2 mesh id2
     * @return whether two mesh specified by ids is covertex
     */
    static bool covertex(int id1, int id2)
    {
        //if (Meshes.empty())
        //    return false;

        //int mid1, fid1, mid2, fid2;

        //mesh_id(id1, Meshes, mid1, fid1);
        //mesh_id(id2, Meshes, mid2, fid2);

        //if (mid1 != mid2)
        //    return false;

        //tri3f& f1 = Meshes[mid1]->_tris[fid1];
        //tri3f& f2 = Meshes[mid2]->_tris[fid2];

        //for (int i = 0; i < 3; i++)
        //    for (int j = 0; j < 3; j++)
        //        if (f1.id(i) == f2.id(2))
        //            return true;

        return false;
    }

    /**
     * storage mesh
     *
     * @param[in] meshes meshes to be storaged
     */
    static void self_mesh(std::vector<triMesh*>& meshes)
    {
        //Meshes = meshes;
    }
};
}  // namespace Physika