#include "mesh.h"
#include <GL/glew.h>
#include <light_array.h>
#include "obj.h"
#include "../util.h"

int make_mesh_dl(Mesh &mesh)
{
	int dl = glGenLists(1);
	glNewList(dl, GL_COMPILE);

	glShadeModel(GL_SMOOTH);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glVertexPointer(3, GL_FLOAT, sizeof(r32) * 8, mesh.vdata);
	glNormalPointer(GL_FLOAT, sizeof(r32) * 8, mesh.vdata + 3);

	glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, mesh.idata);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glEndList();
	return dl;
}

Mesh graphics_mesh_create(Vertex* vertices, u32* indices)
{
	Mesh mesh;

	int num = array_length(indices);
	mesh.num_indices = num;
	mesh.idata = new u32[num];
	memcpy(mesh.idata, indices, sizeof(u32) * num);

	mesh.vdata = new r32[8 * num];
	memcpy(mesh.vdata, vertices, sizeof(u32)*8*num);

	mesh.dl = make_mesh_dl(mesh);

	return mesh;
}

void mesh_destroy(Mesh* mesh) {
	return;//repeated references...

	if (mesh->idata != nullptr) {
		delete[] mesh->idata;
		delete[] mesh->vdata;
		mesh->idata = nullptr;
		mesh->vdata = nullptr;
	}
}


void render_mesh(Mesh& mesh, r32* mat)
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(mat);

	glCallList(mesh.dl);

	glPopMatrix();
}
