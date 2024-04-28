#include <GL/glut.h>
#include <SOIL/SOIL.h>

// Global variables
int screenWidth, screenHeight;
float zoomFactor = 1.0;
int lastMouseX, lastMouseY;
bool mouseLeftDown = false;
GLuint textureID;

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

  // Load and bind texture
  GLuint textureID = SOIL_load_OGL_texture(
      "sample.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_INVERT_Y);
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Enable texture coordinate handling
  glEnable(GL_TEXTURE_2D);

  // Draw rectangle with texture mapping
  double scale = zoomFactor;
  glBegin(GL_POLYGON);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-scale, -scale);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(scale, -scale);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(scale, scale);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-scale, scale);
  glEnd();

  // Disable texture coordinate handling
  glDisable(GL_TEXTURE_2D);

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
