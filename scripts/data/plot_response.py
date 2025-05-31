import matplotlib.pyplot as plt


def plot_palette(response_data):
    # Set up the plot
    fig, ax = plt.subplots(len(response_data['results']), 1, figsize=(10, len(response_data['results']) * 2))
    fig.suptitle("Color Palettes with Scores", fontsize=16)

    # Plot each palette
    for idx, result in enumerate(response_data['results']):
        palette = result['palette']
        score = result['score']
        
        # Create a row for each palette
        for color_idx, color in enumerate(palette):
            ax[idx].add_patch(plt.Rectangle((color_idx, 0), 1, 1, color=color))
        
        # Set axis limits and labels
        ax[idx].set_xlim(0, len(palette))
        ax[idx].set_ylim(0, 1)
        ax[idx].axis("off")
        ax[idx].text(len(palette) + 0.5, 0.5, f"Score: {score:.2f}", va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()