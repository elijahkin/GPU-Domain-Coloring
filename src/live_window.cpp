#include <GL/glut.h>

// Global variables
int screenWidth, screenHeight;
float zoomFactor = 1.0;
int lastMouseX, lastMouseY;
bool mouseLeftDown = false;

void mouse(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_DOWN) {
      mouseLeftDown = true;
      lastMouseX = x;
      lastMouseY = y;
    } else if (state == GLUT_UP) {
      mouseLeftDown = false;
    }
  } else if (button == 3) { // Scroll up
    zoomFactor *= 1.1;
  } else if (button == 4) { // Scroll down
    zoomFactor *= 0.9;
  }
  glutPostRedisplay();
}

void motion(int x, int y) {
  if (mouseLeftDown) {
    int deltaX = x - lastMouseX;
    int deltaY = y - lastMouseY;

    // Update the center of the view based on mouse movement
    float translateX = 2 * static_cast<float>(deltaX) / screenWidth;
    float translateY = -2 * static_cast<float>(deltaY) / screenHeight;

    glTranslatef(translateX, translateY, 0.0f);

    // Update the last mouse position to the current position
    lastMouseX = x;
    lastMouseY = y;

    glutPostRedisplay();
  }
}

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
  case 27: // Escape key
    exit(0);
    break;
  case 32: // Space
    zoomFactor = 1.0;
    glLoadIdentity();
    glutPostRedisplay();
  }
}

void display() {
  // Clear the color buffer
  glClear(GL_COLOR_BUFFER_BIT);

  // Draw rectangle
  double scale = zoomFactor;
  glBegin(GL_POLYGON);
  glColor3f(0, 0, 0);
  glVertex2f(-scale, -scale);
  glColor3f(0, 1, 1);
  glVertex2f(scale, -scale);
  glColor3f(1, 0, 1);
  glVertex2f(scale, scale);
  glColor3f(1, 1, 0);
  glVertex2f(-scale, scale);
  glEnd();

  // Swap the front and back buffers
  glutSwapBuffers();
}

int main(int argc, char **argv) {
  // Initialize GLUT
  glutInit(&argc, argv);

  // Set display mode
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

  // Get screen size
  screenWidth = glutGet(GLUT_SCREEN_WIDTH);
  screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

  // Set up window size and name
  glutInitWindowSize(screenWidth, screenHeight);
  glutCreateWindow("Functiongram");

  // Set fullscreen
  glutFullScreen();

  // Set up the callback functions
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  // Enter the GLUT event processing loop
  glutMainLoop();

  return 0;
}
