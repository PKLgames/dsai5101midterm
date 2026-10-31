import matplotlib.pyplot as plt
import networkx as nx

# Initialize graph
G = nx.Graph()

# Group A nodes (Left) and Group B nodes (Right)
group_a = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
group_b = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']

# Set fixed positions so they form two islands
pos = {
    'A1': (0, 2), 'A2': (0.5, 3), 'A3': (0.5, 1), 'A4': (1, 2.5), 'A5': (1, 1.5), 'A6': (1.5, 2),
    'B1': (5, 2), 'B2': (4.5, 3), 'B3': (4.5, 1), 'B4': (4, 2.5), 'B5': (4, 1.5), 'B6': (3.5, 2)
}

# Dense internal edges for Group A (solid)
edges_a = [
    ('A1', 'A2'), ('A1', 'A3'), ('A1', 'A4'), ('A1', 'A5'),
    ('A2', 'A4'), ('A2', 'A5'), ('A2', 'A6'),
    ('A3', 'A4'), ('A3', 'A5'), ('A3', 'A6'),
    ('A4', 'A6'), ('A5', 'A6'), ('A4', 'A5')
]

# Dense internal edges for Group B (solid)
edges_b = [
    ('B1', 'B2'), ('B1', 'B3'), ('B1', 'B4'), ('B1', 'B5'),
    ('B2', 'B4'), ('B2', 'B5'), ('B2', 'B6'),
    ('B3', 'B4'), ('B3', 'B5'), ('B3', 'B6'),
    ('B4', 'B6'), ('B5', 'B6'), ('B4', 'B5')
]

# ONLY ONE cross edge now (the broken bridge)
edges_cross = [('A6', 'B6')]

# Draw
fig, ax = plt.subplots(figsize=(9, 3.5)) # Made slightly shorter since bottom text is gone
nx.draw_networkx_nodes(G, pos, nodelist=group_a, node_color='#4A90E2', node_size=500, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=group_b, node_color='#F5A623', node_size=500, ax=ax)

# Draw the dense internal edges
nx.draw_networkx_edges(G, pos, edgelist=edges_a, width=1.5, edge_color='#A0A0A0', alpha=0.7, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=edges_b, width=1.5, edge_color='#A0A0A0', alpha=0.7, ax=ax)

# Draw the single sparse cross edge (red dotted)
nx.draw_networkx_edges(G, pos, edgelist=edges_cross, width=2, edge_color='#D32F2F', style='dotted', ax=ax)

# Add Labels
ax.text(0.75, 3.5, "Group A Matchups", fontsize=13, weight='bold', ha='center', color='#333333')
ax.text(4.25, 3.5, "Group B Matchups", fontsize=13, weight='bold', ha='center', color='#333333')

# Middle Text with White Background (Kept this!)
ax.text(2.5, 2.0, "Not enough\nshared votes!", 
        fontsize=14, weight='bold', color='#D32F2F', ha='center', va='center',
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='#D32F2F', boxstyle='round,pad=0.4'))

# Clean up axes
ax.axis('off')
plt.tight_layout()
plt.savefig("two_islands_graph.png", dpi=300)
plt.show()