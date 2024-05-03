#include "complex_plot.cu"
#include <GL/glut.h>

// Global variables
int screenWidth, screenHeight;
float zoomFactor = 1.0;
int lastMouseX, lastMouseY;
bool mouseLeftDown = false;
uint8_t *rgb;
Image render;

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
    cudaFree(rgb);
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

  // Allocate CUDA memory
  cudaMallocManaged(&rgb, screenWidth * screenHeight * 3 * sizeof(uint8_t));
  Image render = {screenWidth, screenHeight, screenWidth * screenHeight, rgb};

  // Rerender TODO move this into display
  domain_color_kernel<<<28, 128>>>([] __device__(Complex z) { return z; },
                                   render, -1, 1, 0.001);
  cudaDeviceSynchronize();

  // Set up window size and name
  glutInitWindowSize(screenWidth, screenHeight);
  glutCreateWindow("Functiongram");

  // Set fullscreen
  glutFullScreen();

  // Set up OpenGL for texture mapping
  glEnable(GL_TEXTURE_2D);

  // Generate a new texture object
  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Set texture parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  // Specify texture image data
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screenWidth, screenHeight, 0, GL_RGB,
               GL_UNSIGNED_BYTE, rgb);

  // Set up the callback functions
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  // Enter the GLUT event processing loop
  glutMainLoop();

  return 0;
}
