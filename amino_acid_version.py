import pygame
import sys
import random
import time
import os
from os import path
import numpy as np

pygame.mixer.init(44100, -16, 2, 2048)

#импорт фоновой музыки
snd_dir = path.join(path.dirname(__file__), 'Music')
pygame.mixer.music.load(path.join(snd_dir, 'orochi_theme.ogg'))

#настройка громкости фоновой музыки
pygame.mixer.music.set_volume(0)

#импорт необходимых классов и функций
from generate_food_pos import generate_food_pos

from food import Food
from snake import Snake
from game import Game

#создание объектов игры

game = Game()
food = Food(game.screen_width, game.screen_height, Snake.dafault_snake_body)
snake = Snake(game, food)

##инициирование дисплея

game.set_display()
game.init_display(game.display)

pygame.mixer.music.play(-1)

### ONAGENDA

# s = env.reset()

# next_s, r, done, _ = env.step(a)


#n_actions = 4
#state_dim = 3 + 3 + 3 + 2 + snakebodylen

#def generate_session(t_max=1000, epsilon=0, train=False):
#    """play env with approximate q-learning agent and train it at the same time"""
#    total_reward = 0
#    s = env.reset() # ([acid1(x, y, goal), acid2, acid3, self.snake_head_pos,
#                        #np.array(self.snake_body + [0 for i in range(maxlen - #len(self.snake_body))]).flatten()])
#
#    for t in range(t_max):
#        a = get_action(s, epsilon=epsilon)
#        next_s, r, done, _ = env.step(a)
#
#        if train:
#            opt.zero_grad()
#            compute_td_loss([s], [a], [r], [next_s], [done]).backward()
#            opt.step()
#
#        total_reward += r
#        s = next_s
#        if done:
#            break
#
#    return total_reward
    

#for i in range(100):
#    session_rewards = [generate_session(
#        epsilon=epsilon, train=True) for _ in range(100)]
##     print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(
##         i, np.mean(session_rewards), epsilon))
#    rewards.append(np.mean(session_rewards))
#    clear_output(True)
#    print('eps =', epsilon, 'mean reward =', np.mean(rewards[-10:]))
#    plt.plot(rewards)
#    plt.show()
#    epsilon *= 0.99
#    assert epsilon >= 1e-4, "Make sure epsilon is always nonzero during training"
#
#    if np.mean(session_rewards) > 300:
#        print("AI wins the game, Snake sucks!")
#        break
#        
    ##########################    
    
#запуск игры
while game.is_alive:
    game.current_display(game.display)
    #total_reward += r
    
    # food.present_food
    # food.food_pos.flatten()
    #a1 = np.array(food.present_food)
    #a2 = food.food_pos.flatten()
    #a3 = np.array(snake.snake_head_pos)
    #a4 = np.array(self.snake_body + [0 for i in range(maxlen - #len(self.snake_body))]).flatten()
    #s = np.concatenate((a1, a2, a3, a4), axis=0)
    #s = [acid1(x, y, type), acid2, acid3, goal_food, self.snake_head_pos,
    #                    np.array(self.snake_body + [0 for i in range(maxlen - #len(self.snake_body))]).flatten()])
    
   
    #пока игра не окончена, будут повторяться следующие действия - env.step. по сути
    if game.game_is_runned:
        snake.proceed_game()

        snake.draw_snake(game.display)
        food.draw_food(game.display)
        game.show_score()

        if not game.game_is_runned:
            food = Food(game.screen_width, game.screen_height, Snake.dafault_snake_body)
            snake = Snake(game, food)
            
    #r, done  = env.step()
    
    #a1 = np.array(food.present_food)
    #a2 = food.food_pos.flatten()
    #a3 = np.array(snake.snake_head_pos)
    #a4 = np.array(self.snake_body + [0 for i in range(maxlen - #len(self.snake_body))]).flatten()        
    #next_s = np.concatenate((a1, a2, a3, a4), axis=0)
    
    ####
    #a = get_action(s, epsilon=epsilon)
    
    #if train:
    #    opt.zero_grad()
    #    compute_td_loss([s], [a], [r], [next_s], [done]).backward()
    #    opt.step()
    #total_reward += r
    #s = next_s
    #if done:
    #    break
    ######
    game.refresh_screen()
