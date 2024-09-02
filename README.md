
# 🤖 Webots Projects Repository

Welcome to the Webots Projects Repository! This repository showcases a variety of projects developed using Webots, a powerful open-source robot simulation software. Here, you'll find implementations of different robotic systems and algorithms, each demonstrating unique aspects of robotics and simulation.


## ✨ Features

Robotic Simulations: Explore various robot models and simulations, including mobile robots, manipulators, and more.
Reinforcement Learning: Examples of reinforcement learning algorithms applied to robotic control, such as Q-learning for obstacle avoidance.
Sensor Integration: Projects demonstrating the use of different sensors (e.g., distance sensors, cameras) within the Webots environment.
## System Requirements

- Official specs

- 2 GHz dual core processor (quad core recommended)

- 8 GB RAM

- NVIDIA or AMD OpenGL graphics adapter with atleast 512 RAM

- Ubuntu, Windows (64 bit only)

## 🌍 Create Your World

The **World file** in Webots defines the layout and properties of objects and the environment (sky, floor, lighting, forces, etc.). Here's how to get started:

### 🛠 Steps to Create a New Project:
- **Create a New Project**:
   - Go to **Wizards** > **New Project Directory**.
   - Name the project directory **`test`**.
   - Name the world file **`test.wbt`**.
   - Check all text boxes and click **Finish**.

- **Explore the Scene Tree**:
   - Hover over nodes to view their descriptions.

- **Modify Fields**:
   - Double-click **RectangleArena** > **Select floorTileSize** > Change value to **`1`**.
   - Modified fields will be highlighted in **different colors**.

---

## 🤖 Add an E-puck Robot

- **Ensure Simulation is Paused**:
   - Confirm the elapsed time is **0**.
   - Follow the **{pause -> reset -> modify -> save}** cycle.

- **Add the E-puck**:
   - Select **WoodenBox** > **Add** > **PROTO nodes** (from Webots Projects) > **robots**.
   - Navigate to **gctronic** > **e-puck** > **E-puck (Robot)** and click **Save**.

---

## 📝 Create a Controller

- **Generate a New Robot Controller**:
   - Go to **Wizards** > **New Robot Controller...**.
   - Choose your **Language**, name it **`test_control`**, and check **Open in text editor**.
   - A new file opens in the text editor window, and an independent Python file and directory are created in **`test/controllers`**.

- **Add Code**:
   - Insert the code from `obstacle_avoidance.py`, which is available on GitHub.

---

## 🚀 Run the Simulation

- Follow the instructions in the project's README file to **run the Webots simulation** and see your creations in action!

---
