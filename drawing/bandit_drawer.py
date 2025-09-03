from model.arm import Arm
import matplotlib.pyplot as plt

# Author: Claude code
def draw(arms: list[Arm], filename: str):
    # Configure matplotlib to match thesis styling with LaTeX rendering
    plt.rcParams.update({
        'text.usetex': True,  # Use LaTeX for text rendering
        'font.family': 'serif',
        'font.serif': ['Palatino'],
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 12,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
    
    # TUM color scheme from thesis
    TUMBlue = '#0065BD'
    TUMAccentOrange = '#E37222'
    TUMAccentGreen = '#A2AD00'
    TUMSecondaryBlue2 = '#003359'
    TUMDarkGray = '#333333'
    
    # Sample 20 values from each arm and plot with different colors and shapes
    # Use larger figure size for better text rendering at scale
    plt.figure(figsize=(5, 5), dpi=150)

    # Get unique clusters to assign colors
    unique_clusters = list(set(arm.cluster for arm in arms))
    # Use TUM color palette
    tum_colors = [TUMBlue, TUMAccentOrange, TUMAccentGreen, TUMSecondaryBlue2, TUMDarkGray]
    cluster_color_map = {cluster: tum_colors[i % len(tum_colors)] for i, cluster in enumerate(unique_clusters)}
    
    # Define different markers for each arm
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8', 'd']

    for i, arm in enumerate(arms):
        values = arm.sample(20)
        marker = markers[i % len(markers)]  # Cycle through markers if more arms than markers
        color = cluster_color_map[arm.cluster]
        
        plt.scatter(values[:, 0], values[:, 1], 
                c=color, 
                marker=marker,
                label=f'Arm {i+1} (Cluster {arm.cluster+1})',
                alpha=0.8,
                s=60,
                edgecolors='white',
                linewidth=0.5)

    plt.xlabel('Dimension 1', fontweight='normal')
    plt.ylabel('Dimension 2', fontweight='normal')
    plt.title('20 Samples from Each Arm', fontweight='normal', pad=20)
    plt.legend(frameon=True, fancybox=False, shadow=False, 
              framealpha=1.0, edgecolor='black', borderpad=0.5)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save as vector graphic PDF for crisp, scalable output
    plt.savefig(filename, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                metadata={'Creator': 'matplotlib', 'Producer': 'matplotlib'})
    plt.show()


# Author: Claude code
def draw_sampling_complexity_comparison(distances, complexities_vkabc_empirical, complexities_kabc, filename, xlabel='Variance Factor'):
    """Draw a comparison plot of empirical VKABC vs KABC sampling complexities"""
    
    # Configure matplotlib to match thesis styling with LaTeX rendering
    plt.rcParams.update({
        'text.usetex': True,  # Use LaTeX for text rendering
        'font.family': 'serif',
        'font.serif': ['Palatino'],
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 12,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
    
    # TUM color scheme
    TUMBlue = '#0065BD'
    TUMAccentOrange = '#E37222'
    
    plt.figure(figsize=(5, 5), dpi=150)
    
    plt.plot(distances, complexities_vkabc_empirical, 'o-', linewidth=2, markersize=8, 
             color=TUMBlue, label='VKABC')
    plt.plot(distances, complexities_kabc, 's-', linewidth=2, markersize=8, 
             color=TUMAccentOrange, label='KABC')
    
    plt.xlabel(xlabel)
    plt.ylabel('Sampling Complexity')
    plt.title('Empirical Sampling Complexity Comparison')
    plt.legend(frameon=True, fancybox=False, shadow=False, 
              framealpha=1.0, edgecolor='black', borderpad=0.5)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Ensure axes start at 0
    plt.xlim(0, None)
    plt.ylim(0, None)
    
    plt.tight_layout()
    
    # Save as vector graphic PDF
    plt.savefig(filename, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                metadata={'Creator': 'matplotlib', 'Producer': 'matplotlib'})
    plt.show()


# Author: Claude code
def draw_theoretical_complexity(distances, complexities_vkabc_empirical, tau_values, filename, factor=None, xlabel='Variance Factor'):
    """Draw comparison of empirical vs theoretical VKABC sampling complexity"""

    if factor:
        complexities_vkabc_empirical = [factor * c for c in complexities_vkabc_empirical]
    
    # Configure matplotlib to match thesis styling with LaTeX rendering
    plt.rcParams.update({
        'text.usetex': True,  # Use LaTeX for text rendering
        'font.family': 'serif',
        'font.serif': ['Palatino'],
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 12,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
    
    # TUM color scheme
    TUMBlue = '#0065BD'
    TUMAccentGreen = '#A2AD00'
    
    plt.figure(figsize=(5, 5), dpi=150)
    
    if factor:
        label = f'{factor} · VKABC (Empirical)'
    else:
        label = 'VKABC (Empirical)'

    plt.plot(distances, complexities_vkabc_empirical, 'o-', linewidth=2, markersize=8, 
             color=TUMBlue, label=label)
    plt.plot(distances, tau_values, '^--', linewidth=2, markersize=8, 
             color=TUMAccentGreen, label='VKABC (Theoretical)')
    
    plt.xlabel(xlabel)
    plt.ylabel('Sampling Complexity')
    plt.title('Empirical vs Theoretical VKABC Complexity')
    plt.legend(frameon=True, fancybox=False, shadow=False, 
              framealpha=1.0, edgecolor='black', borderpad=0.5)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Ensure axes start at 0
    plt.xlim(0, None)
    plt.ylim(0, None)
    
    plt.tight_layout()
    
    # Save as vector graphic PDF
    plt.savefig(filename, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                metadata={'Creator': 'matplotlib', 'Producer': 'matplotlib'})
    plt.show()