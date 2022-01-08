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
# index 10 ='index_move'


class Game:
    def __init__(self):
        self.obs = np.array(
            [10, 10, 630, 470, width/2, height/2, 5, 0, 0, 0, 0])
        self.n_observation = self.obs.shape
        self.n_actions = 4
        self.last_move = 0
        self.total_moves = 1000
        self.current_moves = 0
        self._obs = self.obs = np.array(
            [10/width, 10/height, 630/width, 470/height, (width/2)/width, (height/2)/height, 10, 0, 0, 0, 0])

    def normalize(self):
        self._obs[0] = self.obs[0] / width
        self._obs[1] = self.obs[1] / height
        self._obs[2] = self.obs[2] / width
        self._obs[3] = self.obs[3] / height
        self._obs[4] = self.obs[4] / width
        self._obs[5] = self.obs[5] / height
        self._obs[6] = self.obs[6] / 5
        self._obs[7] = self.obs[7] / 1
        self._obs[8] = self.obs[8] / width
        self._obs[9] = self.obs[9]
        self._obs[10] = self.obs[10]

    def reset(self):
        self.obs = np.array(
            [10, 10, 630, 470, width/2, height/2, 5, 0, 0, 0, 0])
        self.n_observation = self.obs.shape
        self.n_actions = 4
        self.done = False
        self.reward = 0
        self.last_pos = [0, 0]
        self.new_pos = [0, 0]
        self.fps = 24
        self.gettingGo_times = 1
        self.current_moves = 0
        self.isCollision = False
        return self.copyArray(self.obs)

    def copyArray(self, arr):
        return np.copy(arr)

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
            self.reward = -10
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
            self.current_moves = 0
        if self.current_moves > self.total_moves:
            self.done = True
            self.reward = -100

    def step(self, action):
        result = {
            0: self.up,
            1: self.right,
            2: self.down,
            3: self.left,
        }

        for i, act in enumerate(action):
            result[i](act)
        self.last_move = np.argmax(action)
        self.check_distance()
        self.current_moves += 1
        self.obs[10] = self.current_moves/self.total_moves
        self.normalize()
        return self._obs, self.reward, self.done

    def down(self, action):
        speed = self.obs[6]
        if self.last_move == 0:
            return
        if self.obs[5] < height-10:
            if action == 1:
                self.obs[5] += speed
            self.obs[9] = 0
        else:
            self.obs[9] = 1

    def up(self, action):
        if self.last_move == 2:
            return
        speed = self.obs[6]
        if self.obs[5] > 5:
            if action == 1:
                self.obs[5] -= speed
            self.obs[9] = 0
        else:
            self.obs[9] = 1

    def left(self, action):
        if self.last_move == 1:
            return
        if self.obs[4] > 5:
            if action == 1:
                self.obs[4] -= self.obs[6]
            self.obs[9] = 0
        else:
            self.obs[9] = 1

    def right(self, action):
        if self.last_move == 3:
            return
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
        point1x, point1y, point2x, point2y, playerx, playery,\
            speed, index_nextPoint, distance, collision, index_move = self.obs
        circle(screen, (255, 255, 255), (point1x, point1y), 5)
        circle(screen, (255, 255, 255), (point2x, point2y), 5)
        rect(screen, (255, 255, 255), (playerx, playery, 10, 10))
        pygame.display.flip()

    def __str__(self) -> str:
        return str(self.reward)
