#include <GL/glut.h>

float zoomFactor = 1.0;

void display() {
  // Clear the color buffer
  glClear(GL_COLOR_BUFFER_BIT);

  // Draw a red square
  glColor3f(1.0, 0.0, 0.0);
  glBegin(GL_POLYGON);

  double scale = 0.5 * zoomFactor;

  glVertex2f(-scale, -scale);
  glVertex2f(scale, -scale);
  glVertex2f(scale, scale);
  glVertex2f(-scale, scale);
  glEnd();

  // Swap the front and back buffers
  glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
  case 27: // Escape key
    exit(0);
    break;
  case 'w':
    zoomFactor *= 1.1;
    glutPostRedisplay();
    break;
  case 's':
    zoomFactor *= 0.9;
    glutPostRedisplay();
    break;
  }
}

int main(int argc, char **argv) {
  // Initialize GLUT
  glutInit(&argc, argv);

  // Set display mode
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

  // Set up window size and name
  glutInitWindowSize(600, 400);
  glutCreateWindow("Functiongram");

  // Set up the callback functions
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  // glutMouseWheelFunc(mouseWheel);

  // Enter the GLUT event processing loop
  glutMainLoop();

  return 0;
}
