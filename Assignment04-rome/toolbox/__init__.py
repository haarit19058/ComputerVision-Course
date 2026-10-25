"""The toolbox: everything in "Rome in a Day: A City Hidden in Photos" that you do not write yourself.

    from toolbox import *      (the notebook does this for you)

Every function has a docstring: help(verify_pairs). The modules, in the order the notebook uses them:
  data      the photo collection: photos, hidden labels, survey cameras, visual words
  graph     UnionFind, retrieval pairs, pair checks, components, the maximum spanning tree
  geometry  two view geometry, triangulation (for the pipeline), PnP, bundle adjustment (one K per photo)
  sfm       one group of photos: SIFT, tracks, the skeleton SfM, everyone else by PnP, the St Peter's surveys
  dense     the dense cloud
  plots     every picture
  checks    test_<name>() for your 9 functions, with small synthetic data and edge cases
  reference ref.<name>: the library version of each of your functions, same inputs and outputs

Your 9 functions are not in here: the toolbox looks them up in the notebook when it needs them (see _yours.py).

Conventions used everywhere: photos are numbered 0 .. N-1; a pair or an edge is a row (i, j) of an (M, 2) int64 array
with i < j; pixels are (N,2) arrays (x to the right, y down); K is a 3x3 intrinsic matrix, one per photo; a camera
pose (R, t) maps world to camera, x ~ K (R X + t), P = K [R | t], t has shape (3,); the camera centre is C = -R^T t.
"""

from . import style  # noqa: F401  (sets the picture style)
from .data import (pretty, load_mini_rome, load_image, thumbnail, label_name)
from .geometry import (to_h, skew, nearest_rotation, rodrigues, rodrigues_inv, projection_matrix, project,
                       reprojection_error, camera_center, pixel_colors, ray_angles, hartley_normalize,
                       essential_from_F, fundamental_from_poses, decompose_E, eight_point, epipolar_distance,
                       ransac_fundamental, ransac_fundamental_cv, choose_pose, pnp, refine_pose,
                       refine_relative_pose, pnp_ransac, pack_params, unpack_params, reprojection_residuals,
                       jacobian_sparsity, bundle_adjust)
from .graph import (UnionFind, neighbour_pairs, retrieval_precision, word_matches, verify_pairs, component_purity,
                    photos_of, component_of_landmark, component_graph, maximum_spanning_tree, skeleton_nodes)
from .sfm import (sift_features, landmark_scene, scene_image, pair_matches, pair_cameras, triangulation_pair,
                  model_pair, strongest_pairs, run_sfm_skeleton, register_rest, camera_centres_vs_gt, st_peters_maps)
from .dense import (clean_cloud, densify, survey_scene, densify_survey)
from .plots import (show_thumbs, show_crate, show_words, plot_similarity_matrix, show_neighbours,
                    plot_retrieval_precision, show_matches, show_word_matches, show_match_filters,
                    show_homography_test, show_reprojection, baseline_experiment, compare_with_model,
                    graph_layout, plot_match_graph,
                    show_components, plot_spanning_tree, plot_reconstruction, plot_alignment, plot_merge, show_atlas,
                    plot_dense_merge)
from .checks import (make_bags, make_graph, make_point_pairs, make_descriptors, make_two_views, make_triangulation,
                     check,
                     test_tfidf_matrix, test_top_k_neighbours, test_match_descriptors, test_query_expansion,
                     test_connected_components, test_homography_dlt, test_homography_ratio, test_triangulate_points,
                     test_umeyama, run_all_tests)
from .reference import ref

__all__ = [
    "pretty", "load_mini_rome", "load_image", "thumbnail", "label_name", "to_h", "skew", "nearest_rotation",
    "rodrigues", "rodrigues_inv", "projection_matrix", "project", "reprojection_error", "camera_center",
    "pixel_colors", "ray_angles", "hartley_normalize", "essential_from_F", "fundamental_from_poses", "decompose_E",
    "eight_point", "epipolar_distance", "ransac_fundamental", "ransac_fundamental_cv", "choose_pose",
    "pnp", "refine_pose", "refine_relative_pose", "pnp_ransac", "pack_params", "unpack_params",
    "reprojection_residuals", "jacobian_sparsity", "bundle_adjust", "UnionFind", "neighbour_pairs",
    "retrieval_precision", "word_matches", "verify_pairs", "component_purity", "photos_of", "component_of_landmark",
    "component_graph", "maximum_spanning_tree", "skeleton_nodes", "sift_features", "landmark_scene", "scene_image",
    "pair_matches", "pair_cameras", "triangulation_pair", "model_pair",
    "strongest_pairs", "run_sfm_skeleton", "register_rest",
    "camera_centres_vs_gt", "st_peters_maps", "clean_cloud", "densify", "survey_scene", "densify_survey", "show_thumbs", "show_crate", "show_words",
    "plot_similarity_matrix", "show_neighbours", "plot_retrieval_precision", "show_matches", "show_word_matches",
    "show_match_filters", "show_homography_test", "show_reprojection", "baseline_experiment", "compare_with_model",
    "graph_layout", "plot_match_graph", "show_components", "plot_spanning_tree", "plot_reconstruction",
    "plot_alignment", "plot_merge", "show_atlas", "plot_dense_merge", "make_bags", "make_graph", "make_point_pairs",
    "make_descriptors", "make_two_views", "make_triangulation", "check", "test_tfidf_matrix", "test_top_k_neighbours",
    "test_match_descriptors", "test_query_expansion", "test_connected_components", "test_homography_dlt",
    "test_homography_ratio", "test_triangulate_points", "test_umeyama", "run_all_tests", "ref",
]

# written by dev/build.py: a hash of the toolbox sources; the notebook checks it before importing
TOOLBOX_VERSION = "63efbeb30581"
