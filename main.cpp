#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <random>
#include "shader_utils.hpp"

const int PARTICLE_COUNT = 2;
const float G = 1.0f;
const float SOFTENING = 0.01f;
const float RADIUS_SCALE = 0.02f;

struct Particle {
    glm::vec2 pos;
    glm::vec2 vel;
    float mass;
};

struct GPUData {
    glm::vec2 pos;
    float mass;
};

std::vector<Particle> particles;

float getRadius(float mass) {
    return mass * RADIUS_SCALE;
}

const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aMass;
out float mass;

void main() {
    mass = aMass;
    gl_PointSize = aMass * 10.0; // scale visual size
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in float mass;
out vec4 FragColor;

void main() {
    float r = length(gl_PointCoord - vec2(0.5));
    if (r > 0.5) discard;

    vec3 base = vec3(0.1, 0.6, 1.0);
    vec3 heavy = vec3(1.0, 0.9, 0.2);
    vec3 color = mix(base, heavy, clamp(mass / 2.0, 0.0, 1.0));
    float alpha = 1.0 - smoothstep(0.45, 0.5, r);
    FragColor = vec4(color, alpha);
}
)";

void initParticles() {
    std::default_random_engine rng;
    std::uniform_real_distribution<float> xDist(-0.8f, 0.8f);
    std::uniform_real_distribution<float> yDist(-0.8f, 0.8f);
    std::uniform_real_distribution<float> mDist(0.2f, 2.0f);

    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        Particle p;
        p.pos = glm::vec2(xDist(rng), yDist(rng));
        p.vel = glm::vec2(0.0f, 0.0f);
        p.mass = mDist(rng);
        particles.push_back(p);
    }
}

int main() {
    // OpenGL Context Setup
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Gravity + Collisions", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    glewInit();

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint shader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    initParticles();

    // Setup GPU buffer
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, PARTICLE_COUNT * sizeof(GPUData), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GPUData), (void*)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GPUData), (void*)offsetof(GPUData, mass));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        float dt = currentTime - lastTime;
        lastTime = currentTime;

        // Gravitational Forces
        std::vector<glm::vec2> acc(particles.size(), glm::vec2(0.0f));
        for (int i = 0; i < particles.size(); ++i) {
            for (int j = 0; j < particles.size(); ++j) {
                if (i == j) continue;

                glm::vec2 diff = particles[j].pos - particles[i].pos;
                float distSq = glm::dot(diff, diff) + SOFTENING;
                float dist = sqrt(distSq);
                float force = G * particles[i].mass * particles[j].mass / distSq;
                glm::vec2 dir = diff / dist;
                acc[i] += dir * (force / particles[i].mass);
            }
        }

        // Apply acceleration and velocity
        for (int i = 0; i < particles.size(); ++i) {
            particles[i].vel += acc[i] * dt;
        }

        // Collision Detection + Resolution
        for (int i = 0; i < particles.size(); ++i) {
            for (int j = i + 1; j < particles.size(); ++j) {
                glm::vec2 diff = particles[j].pos - particles[i].pos;
                float distSq = glm::dot(diff, diff);
                float rSum = getRadius(particles[i].mass) + getRadius(particles[j].mass);

                if (distSq < rSum * rSum) {
                    float dist = sqrt(distSq);
                    if (dist < 0.0001f) continue; // avoid NaNs

                    glm::vec2 n = diff / dist;
                    float v1 = glm::dot(particles[i].vel, n);
                    float v2 = glm::dot(particles[j].vel, n);
                    float m1 = particles[i].mass;
                    float m2 = particles[j].mass;

                    float v1New = (v1 * (m1 - m2) + 2 * m2 * v2) / (m1 + m2);
                    float v2New = (v2 * (m2 - m1) + 2 * m1 * v1) / (m1 + m2);

                    particles[i].vel += (v1New - v1) * n;
                    particles[j].vel += (v2New - v2) * n;

                    // Resolve overlap
                    float penetration = rSum - dist;
                    glm::vec2 correction = n * (penetration / 2.0f);
                    particles[i].pos -= correction;
                    particles[j].pos += correction;
                }
            }
        }

        // Update positions
        for (auto& p : particles) {
            p.pos += p.vel * dt;
        }

        // Update GPU buffer
        std::vector<GPUData> gpuData;
        for (auto& p : particles) {
            gpuData.push_back({ p.pos, p.mass });
        }
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, gpuData.size() * sizeof(GPUData), gpuData.data());

        // Render
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, particles.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
