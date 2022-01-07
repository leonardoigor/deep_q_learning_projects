from typing import Match
import pygame
from pygame.draw import *
import numpy as np

width, height = 640, 480
tick = pygame.time.Clock()
screen = pygame.display.set_mode((width, height))


isRunning = True

# indexs 0,1='first point'
# indexs 2,3='secondd point'
# indexs 4,5='player position'
# index 6 ='speed'
# index 7 ='next point target'
# index 8 ='distance to next point'
# index 9 ='collision'


class Game:
    def reset(self):
        self.obs = [10, 10, 630, 470, 50, 50, 1, 0, 0, 0]
        self.done = False
        self.reward = 0
        self.last_pos = [0, 0]
        self.new_pos = [0, 0]
        self.fps = 24
        self.gettingGo_times = 1
        self.isCollision = False

    def check_distance(self):
        self.new_pos = [self.obs[4], self.obs[5]]
        target = self.obs[7]
        targets = {
            0: [self.obs[0], self.obs[1]],
            1: [self.obs[2], self.obs[3]],
        }
        target_x, target_y = targets[target]

        last_distance = self.getDistanceBetweenPoints(
            target_x, target_y, self.last_pos[0], self.last_pos[1]
        )
        new_distance = self.getDistanceBetweenPoints(
            target_x, target_y,
            self.new_pos[0], self.new_pos[1])
        if new_distance < last_distance:
            self.reward = 1
            self.last_pos = self.new_pos
        if new_distance > last_distance:
            self.reward = -1
            self.last_pos = self.new_pos
        self.obs[8] = new_distance

        collision = self.obs[9]
        if collision == 1:
            self.reward = -10
        if new_distance < 10:
            nIndex = self.gettingGo_times % 2
            self.obs[7] = nIndex
            self.gettingGo_times += 1
            self.reward = 100
            self.done = True

    def step(self, action):
        result = {
            0: self.up,
            1: self.right,
            2: self.down,
            3: self.left,
        }

        for i, act in enumerate(action):
            result[i](act)
        r = np.random.random()
        if r > .99999:
            self.done = True
        self.check_distance()
        return self.obs, self.reward, self.done

    def down(self, action):
        speed = self.obs[6]
        if self.obs[5] < height-10:
            if action == 1:
                self.obs[5] += speed
            self.obs[9] = 0
        else:
            self.obs[9] = 1

    def up(self, action):
        speed = self.obs[6]
        if self.obs[5] > 5:
            if action == 1:
                self.obs[5] -= speed
            self.obs[9] = 0
        else:
            self.obs[9] = 1

    def left(self, action):
        if self.obs[4] > 5:
            if action == 1:
                self.obs[4] -= self.obs[6]
            self.obs[9] = 0
        else:
            self.obs[9] = 1

    def right(self, action):
        if self.obs[4] < width-10:
            if action == 1:
                self.obs[4] += self.obs[6]
            self.obs[9] = 0
        else:
            self.obs[9] = 1

    def getDistanceBetweenPoints(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def draw(self):
        tick.tick(self.fps)
        pygame.event.get()
        screen.fill((0, 0, 0))
        point1x, point1y, point2x, point2y, playerx, playery, speed, index_nextPoint, distance, collision = self.obs
        circle(screen, (255, 255, 255), (point1x, point1y), 5)
        circle(screen, (255, 255, 255), (point2x, point2y), 5)
        rect(screen, (255, 255, 255), (playerx, playery, 10, 10))
        pygame.display.flip()

    def __str__(self) -> str:
        return str(self.reward)
