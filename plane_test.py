import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
#
# post_ply = o3d.io.read_point_cloud("post_office2.ply")
# downpcd = post_ply.voxel_down_sample(voxel_size=0.01)
#
# # plane_model2, inliers2 = post_ply.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
# #
# # [a_test, b_test, c_test, d_test] = plane_model2
# #
# #
# # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
#
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as context_manager:
#     labels = np.array(
#         downpcd.cluster_dbscan(eps=0.001, min_points=100, print_progress=True))
#
# max_label = labels.max()
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
#
# print('labels: ', labels)
# downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([downpcd],)
#
# # o3d.visualization.draw_geometries([post_ply])
# # o3d.visualization.draw_geometries([pcd_test, translated_test, arrow_test_copied, coordinate])
pcd = o3d.io.read_point_cloud(".post_office2.ply")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])