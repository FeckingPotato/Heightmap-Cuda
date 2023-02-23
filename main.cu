#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include "mapGen.cuh"
#include <fstream>
#include <iostream>

const static long size = 2048;
const static int seed = 567;

float Y_AXIS = 0.0;

float* vertices;

void drawScene() {
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0,0.0,-10.0);
    glRotatef(Y_AXIS,0.0,1.0,0.0);
    glScalef(3.0, 3.0, 3.0);
    glPushMatrix();
    glBegin(GL_POINTS);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            glColor3f((float) i/size, vertices[i+size*j], (float) j/size);
            glVertex3f((float) i/size, vertices[i+size*j], (float) j/size);
        }
    }
    glEnd();
    glPopMatrix();
    glutSwapBuffers();
    Y_AXIS += 0.50;
}

void handleResize(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)width/(double)height, 1.0, 200.0);
}

int main(int argc, char *argv[]) {
    cudaMallocManaged(&vertices, size * size * sizeof(float));
    mapGen(vertices, size, seed, 1);
    std::ofstream file;
    file.open("heightmap.pgm");
    file << "P5 " << size << " " << size << " 255" << std::endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            char temp = vertices[i*size+j]*255;
            file.write(&temp, 1);
        }
    }
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
    glutInitWindowSize(600,600);
    glutCreateWindow("Heightmap");
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(reinterpret_cast<void (*)()>(drawScene));
    glutIdleFunc(reinterpret_cast<void (*)()>(drawScene));
    glutReshapeFunc(handleResize);
    glutMainLoop();
    cudaFree(vertices);
    return 0;
}
