import os

import imageio


class Visuals:
    @staticmethod
    def make_gif(folder, gif_name):
        """
        Iterates through all the pngs in the folder that are named <method>_<epsiode>.png
        and creates a gif out of them
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
    def plot_value(action_value_file, episode_count):
        """Making the plots of the value function"""
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
        ax.set_title(f"Ep. {episode_count} action-value function")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
        fig.savefig(os.path.splitext(action_value_file)[0] + ".png")

    @staticmethod
    def plot_win_rates(win_rates_file):
        """Making the plot of the win rate function"""
        with open(win_rates_file, "rb") as f:
            win_rates = np.load(f)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(win_rates)
        ax.set_xlabel("Thousand episode played")
        ax.set_ylabel("Win rate")
        fig.savefig(os.path.splitext(win_rates_file)[0] + ".png")
