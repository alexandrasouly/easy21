import os
import numpy as np
import imageio
import matplotlib.pyplot as plt


class Visuals:
    @staticmethod
    def make_gif(folder, gif_name):
        """
        Iterates through all the pngs in the folder that are named <method>_<episode>.png
        and creates a gif out of them.
        Assumes the pngs are named as "<text>_<text>_<episode number>.png"
        """
        filelist = [
            file
            for file in os.listdir(folder)
            if file.endswith(".png") and "win_rate" not in file
        ]
        images = []
        for filename in sorted(
            filelist, key=lambda a: int(a.split("_")[2].split(".")[0])
        ):
            images.append(imageio.imread(f"{folder}/{filename}"))
        imageio.mimsave(f"{folder}/{gif_name}.gif", images, fps=1)

    @staticmethod
    def plot_value(action_value_file, episode_count, lamda=None):
        """Making the plots of the value function to turn into gifs later"""
        with open(action_value_file, "rb") as f:
            action_value = np.load(f)

        value = np.max(action_value, axis=2)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection="3d")
        x_range = np.arange(1, action_value.shape[0])
        y_range = np.arange(1, action_value.shape[1])
        X, Y = np.meshgrid(x_range, y_range)
        Z = value[X, Y]
        ax.set_xlabel("Dealer First card")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        if lamda == None:
            ax.set_title(f"Ep. {episode_count} action-value function")
        else:
            ax.set_title(f"Ep. {episode_count} action-value function with λ={lamda}")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
        fig.savefig(os.path.splitext(action_value_file)[0] + ".png")

    @staticmethod
    def plot_win_rates(win_rates_file, lamda=None):
        """Making the plot of the win rate function against episodes played"""
        with open(win_rates_file, "rb") as f:
            win_rates = np.load(f)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(win_rates)
        ax.set_xlabel("Thousand episode played")
        ax.set_ylabel("Win rate")
        if lamda == None:
            ax.set_title(f"Win rates")
        else:
            ax.set_title(f"Win rates with λ={lamda}")
        fig.savefig(os.path.splitext(win_rates_file)[0] + ".png")

    @staticmethod
    def plot_mse_learning(mse_file, lamda):
        """Making the plot of the mse against episodes played"""
        with open(mse_file, "rb") as f:
            mse = np.load(f)
        fig0, ax0 = plt.subplots(figsize=(10, 5))
        ax0.plot(mse)
        ax0.set_xlabel("Episode")
        ax0.set_ylabel("MSE")
        ax0.set_title(f"MSE with λ={lamda}")
        fig0.savefig(os.path.splitext(mse_file)[0] + ".png")

    @staticmethod
    def plot_mse_for_lamdas(mse_dict, folder):
        """Making the plot of the win rate function against different lambdas"""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(mse_dict.keys(), mse_dict.values())
        ax.set_xlabel("Lambda")
        ax.set_ylabel("MSE after 10000 episodes")
        ax.set_title(f"MSE for different lambdas")
        fig.savefig(f"{folder}/sarsa_mse_for_lambdas.png")
