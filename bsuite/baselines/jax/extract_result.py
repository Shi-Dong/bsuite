# -*- mode: python; python-indent-offset: 2 -*-
""" Extract and display bsuite experiment (bootstrapped DQN) results. """

import os
import sys
import csv
from termcolor import colored
from absl import app
from absl import flags

flags.DEFINE_string("env_name", "deep_sea_stochastic", "name of the environment")
flags.DEFINE_integer("ensemble", 5, "number of ensembles")

FLAGS = flags.FLAGS


def average_runs(result_dir: str, run_limit: int=100):
  if result_dir[-1] != "/":
    result_dir += "/"

  result_dict = {}
  for i in range(1, run_limit):
    leaf_dir = result_dir + f"run_{i}/"
    if not os.path.isdir(leaf_dir):
      break

    result_files = os.listdir(leaf_dir)
    for result_file in result_files:
      if result_file.startswith("bsuite_id"):
        env_number = int(result_file[:-4].split("-")[-1]) # extract the env numbering from the cvs file name
        result_sub_dict = result_dict.get(env_number, {})
        with open(leaf_dir + result_file, newline='') as csvfile:
          reader = csv.DictReader(csvfile)
          last_field_name = reader.fieldnames[-1]
          row = list(reader)[-1]
          result_sub_dict[last_field_name] = result_sub_dict.get(last_field_name, []) + [round(float(row[last_field_name]),1)]
          result_sub_dict["episode"] = result_sub_dict.get("episode", []) + [int(row["episode"])]
        result_dict[env_number] = result_sub_dict

  return result_dict

def print_result_dict(result_dict):
  ret = ""
  colors = ["red", "yellow", "green", "blue", "cyan", "magenta"]
  for key in result_dict:
    ret += colored(key, attrs=["reverse"]) + "\n"
    i = 0
    for env_num in sorted(list(result_dict[key].keys())):
      for field_name in result_dict[key][env_num].keys():
        if field_name != "episode":
          last_field_name = field_name
      env_returns = result_dict[key][env_num][last_field_name]
      env_episodes = result_dict[key][env_num]["episode"]
      ret += "  " + colored(f"{FLAGS.env_name}/{env_num}:", attrs=["underline"]) + "\n" 
      ret += f"    {last_field_name}: {env_returns}\n"
      ret += f"    episode: {env_episodes}\n"
      N = len(env_returns)  # number of total runs
      completed_returns = [env_returns[i] for i in range(N) if env_episodes[i] == max(env_episodes)]
      ret += f"    average {last_field_name} of {len(completed_returns)} completed runs: " +  colored(f"{round(sum(completed_returns)/len(completed_returns),1)}", colors[i%6]) + f"\n"
      i += 1
  return ret 


def main(_):
  base_dir = "/home/shidong/tmp/bsuite_results"
  env_name = FLAGS.env_name
  ensemble = FLAGS.ensemble
  base_dir += "/" + FLAGS.env_name + f"/ensemble_{ensemble}"
  assert os.path.isdir(base_dir), "Directory doesn't exist:\n" + base_dir 

  result_total_dict = {}
  if os.path.isdir(base_dir + "/baseline"):
    # baseline results
    result_dir = base_dir + "/baseline"
    result_dict = average_runs(result_dir)
    result_total_dict["baseline"] = result_dict 

  for update_period_dir in os.listdir(base_dir):
    if not update_period_dir.startswith("update_period"):
      continue
    key = "update_period = " + update_period_dir.split("_")[-1] + ", "
    for learning_rate_dir in os.listdir(base_dir + "/" + update_period_dir):
      key1 = key + "learning_rate = " + learning_rate_dir.split("_")[-1] + ", "
      for penalty_weight_dir in os.listdir(base_dir + "/" + update_period_dir + "/" + learning_rate_dir):
        key2 = key1 + "penalty_weight = " + penalty_weight_dir.split("_")[-1]
        result_dir = base_dir + "/" + update_period_dir + "/" + learning_rate_dir + "/" + penalty_weight_dir
        result_total_dict[key2] = average_runs(result_dir)

  print(print_result_dict(result_total_dict))



if __name__ == '__main__':
  app.run(main)
