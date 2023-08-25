#conda activate env_open3d_09

import open3d as o3d
import numpy as np
import copy

mesh = o3d.io.read_triangle_mesh("/mnt/petrelfs/caoziang/3D_generation/Checkpoint_all/diffusion_omni_new_v1_noinit0_3/ddpm_5000plytest1/test/0.obj")
vert = np.asarray(mesh.vertices)
min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
# for _ in range(30):
#     cube = o3d.geometry.TriangleMesh.create_box() # 生成小立方体作为噪声添加到mesh
#     cube.scale(0.005, center=cube.get_center()) # 立方体参数
#     cube.translate(
#         (
#             np.random.uniform(min_vert[0], max_vert[0]),
#             np.random.uniform(min_vert[1], max_vert[1]),
#             np.random.uniform(min_vert[2], max_vert[2]),
#         ),
#         relative=False,
#     )
#     mesh += cube
mesh.compute_vertex_normals()
# print("Show input mesh")
# o3d.visualization.draw_geometries([mesh])

print("Cluster connected triangles")

triangle_clusters, cluster_n_triangles, cluster_area = ( mesh.cluster_connected_triangles())  #将每个三角形分配给一组连通的三角形
triangle_clusters = np.asarray(triangle_clusters) #返回每个三角形集群的索引
cluster_n_triangles = np.asarray(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)

print("Show mesh with small clusters removed")
mesh_0 = copy.deepcopy(mesh)
triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
mesh_0.remove_triangles_by_mask(triangles_to_remove)
# o3d.visualization.draw_geometries([mesh_0])
mesh_0.triangle_normals = o3d.utility.Vector3dVector([])
o3d.io.write_triangle_mesh('c.obj',mesh_0)

print("Show largest cluster")
mesh_1 = copy.deepcopy(mesh)
largest_cluster_idx = cluster_n_triangles.argmax()
triangles_to_remove = triangle_clusters != largest_cluster_idx
mesh_1.remove_triangles_by_mask(triangles_to_remove)

mesh_1.triangle_normals = o3d.utility.Vector3dVector([])
o3d.io.write_triangle_mesh('b.obj',mesh_1)
# o3d.visualization.draw_geometries([mesh_1])
