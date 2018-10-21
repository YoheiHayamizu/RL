import os
import time
import sys
import pygame
from pygame.locals import *
import numpy as np

import myUtil
import const

CS = const.CS  # セルのサイズ
NUM_X = const.NUM_X  # セルの個数(列数)
NUM_Y = const.NUM_Y  # セルの個数(行数)
SCR_X = const.SCR_X  # スクリーンサイズ
SCR_Y = const.SCR_Y  # スクリーンサイズ
SCR_RECT = Rect(0, 0, SCR_X, SCR_Y)

START = 0
GOAL = 1
ROAD = 2
WALL = 3

MAIN_REWARD = 100
FIELD1 = const.FIELD2


class Map:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCR_RECT.size)
        pygame.display.set_caption(u"Q-Learning")
        self.font = pygame.font.SysFont("timesnewroman", 42
                                        )
        self.rewardMap = [[0 for i in range(NUM_X)] for j in range(NUM_Y)]
        self.agent = [[0 for i in range(NUM_X)] for j in range(NUM_Y)]

        self.generation = 0
        self.run = False
        self.cursor = [NUM_X / 2, NUM_Y / 2]
        self.clear()
        self.draw(self.screen)
        # clock = pygame.time.Clock()

    # init
    def clear(self):
        self.generation = 0
        for y in range(NUM_Y):
            for x in range(NUM_X):
                if FIELD1[y][x] == GOAL:
                    self.rewardMap[y][x] = MAIN_REWARD
                if FIELD1[y][x] == START:
                    self.agent[y][x] = 1

    def draw(self, screen):
        # print("debug draw")
        for y in range(NUM_Y):
            for x in range(NUM_X):
                if FIELD1[y][x] == WALL:
                    pygame.draw.rect(screen, (0, 0, 0), Rect(x * CS, y * CS, CS, CS))
                elif FIELD1[y][x] == ROAD:
                    pygame.draw.rect(screen, const.CS_COLOR, Rect(x * CS, y * CS, CS, CS))
                elif FIELD1[y][x] == GOAL:
                    pygame.draw.rect(screen, (100, 255, 255), Rect(x * CS, y * CS, CS, CS))
                if self.agent[y][x] == 1:
                    pygame.draw.rect(screen, (0, 0, 255), Rect(x * CS, y * CS, CS, CS))
                pygame.draw.rect(screen, (50, 50, 50), Rect(x * CS, y * CS, CS, CS), 1)
        pygame.draw.rect(screen, (0, 255, 0), Rect(self.cursor[0] * CS, self.cursor[1] * CS, CS, CS), 5)

    def step(self, action):
        y, x = self.get_position(self.agent)
        self.set_position(y, x, action)
        _reward = self.rewardMap[y][x]
        if FIELD1[y][x] == GOAL:
            _done = True
        else:
            _done = False
        _observation = self
        return _observation, _done, _reward

    @staticmethod
    def get_position(array):
        pos = array
        pos = np.array(pos)
        y, x = np.where(pos == 1)
        y = y[0]
        x = x[0]
        # print(y, x)
        return y, x

    def set_position(self, y, x, action):
        noWall = [ROAD, START, GOAL]
        if FIELD1[y][x - 1] in noWall and action == const.LEFT:
            self.agent[y][x] = 0
            self.agent[y][x - 1] = 1
            print('LEFT')
        if FIELD1[y][x + 1] in noWall and action == const.RIGHT:
            self.agent[y][x] = 0
            self.agent[y][x + 1] = 1
            print('RIGHT')
        if FIELD1[y - 1][x] in noWall and action == const.UP:
            self.agent[y][x] = 0
            self.agent[y - 1][x] = 1
            print('UP')
        if FIELD1[y + 1][x] in noWall and action == const.DOWN:
            self.agent[y][x] = 0
            self.agent[y + 1][x] = 1
            print('DOWN')

    def render(self):
        # clock.tick(100)
        self.draw(self.screen)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == K_s:
                    self.run = not self.run
                elif event.key == K_n:
                    self.step(np.random.randint(0, 4))
                elif event.key == K_LEFT:
                    self.cursor[0] -= 1
                    if self.cursor[0] < 0:
                        self.cursor[0] = 0
                elif event.key == K_RIGHT:
                    self.cursor[0] += 1
                    if self.cursor[0] > NUM_X - 1:
                        self.cursor[0] = NUM_X - 1
                elif event.key == K_UP:
                    self.cursor[1] -= 1
                    if self.cursor[1] < 0:
                        self.cursor[1] = 0
                elif event.key == K_DOWN:
                    self.cursor[1] += 1
                    if self.cursor[1] > NUM_Y - 1:
                        self.cursor[1] = NUM_Y - 1
                elif event.key == K_SPACE:
                    x, y = self.cursor
                    print('-----------------------------------')
                    print('      %05.2f' % self.agent[y][x])
                    print('-----------------------------------')


if __name__ == '__main__':
    a_map = Map()
    target_path = './images/'
    # target_path = myUtil.make_dir(target_path, time=True, numbering=True)
    count = 0
    while True:
        # print(a_map.get_position())
        # print(a_map.agent)
        observation, done, reward = a_map.step(np.random.randint(0, 4))
        a_map.render()
        # pygame.image.save(a_map.screen, target_path + const.IMG_FILE + '{0:00d}'.format(count) + const.EXTENSION)
        if done:
            break
        time.sleep(0.01)
        count += 1
