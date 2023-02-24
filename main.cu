#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <string>
#include "mapGen.cuh"

const static long size = 512;
const static int seed = 5617;

float* vertices;
int gridPosX = 0;
int gridPosY = 0;

void drawOverlay() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    int w = glutGet( GLUT_WINDOW_WIDTH );
    int h = glutGet( GLUT_WINDOW_HEIGHT );
    glOrtho( 0, w, 0, h, -1, 1 );
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_DEPTH_TEST );
    glColor3f(1, 0, 0);
    glRasterPos2i(20, 20);
    void *font = GLUT_BITMAP_HELVETICA_18;
    std::string text = "X position: ";
    text += std::to_string(gridPosX);
    text += "\n Y position: ";
    text += std::to_string(gridPosY);
    for (char &c : text)
    {
        glutBitmapCharacter(font, c);
    }
    glEnable (GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

}

void drawScene() {
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-1.0,0.0,-8.0);
    glRotatef(90.0,0.0,1.0,0.0);
    glScalef(3.0, 3.0, 3.0);
    glPushMatrix();
    glBegin(GL_POINTS);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            glColor3f((float) (i)/size, vertices[i + size * j], (float) (j) / size);
            glVertex3f((float) (i)/size, vertices[i + size * j], (float) (j) / size);
        }
    }
    glEnd();
    glPopMatrix();
    drawOverlay();
    glutSwapBuffers();
}

void move(int x, int y) {
    if (gridPosX + x >= 0 && gridPosY + y >= 0){
        gridPosX += x;
        gridPosY += y;
        mapGen(vertices, size, gridPosX, gridPosY, seed, 1);
        cudaDeviceSynchronize();
    }
    drawScene();
}
void handleResize(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)width/(double)height, 1.0, 200.0);
}

void input (unsigned char key, int x, int y) {
    switch (key) {
        case 'w':
            move(0, 1);
            break;
        case 'a':
            move(-1, 0);
            break;
        case 's':
            move(0, -1);
            break;
        case 'd':
            move(1, 0);
            break;
        case 'W':
            move(0, 10);
            break;
        case 'A':
            move(-10, 0);
            break;
        case 'S':
            move(0, -10);
            break;
        case 'D':
            move(10, 0);
            break;
    }
}

int main(int argc, char *argv[]) {
    cudaMallocManaged(&vertices, size*size * sizeof(float));
    mapGen(vertices, size, gridPosX, gridPosY, seed, 1);
    cudaDeviceSynchronize();
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
    glutInitWindowSize(800,600);
    glutCreateWindow("Heightmap");
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(reinterpret_cast<void (*)()>(drawScene));
    glutReshapeFunc(handleResize);
    glutKeyboardFunc(input);
    glutMainLoop();
    cudaFree(vertices);
    return 0;
}
