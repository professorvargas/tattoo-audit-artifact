# VLM Panels Export

## image_id: `4954893280_86e164e92f_b` (test_open)
**GT (image-level):** `eagle`

**Crop label:** `eagle.png`

| Model | Baseline (pred / fp / n_fp) | Crop black (pred / fp / n_fp) | Crop white (pred / fp / n_fp) |
|---|---|---|---|
| `gemma3` | `branch,eagle,unknown`<br>fp:`branch;unknown`<br>n_fp:`2` | `octopus;unknown;water`<br>fp:`octopus;unknown;water`<br>n_fp:`3` | `octopus;snake;unknown;water`<br>fp:`octopus;snake;unknown;water`<br>n_fp:`4` |
| `llama3_2_vision` | `unknown`<br>fp:`unknown`<br>n_fp:`1` | `anchor,bird,cat,crown,diamond,dog,eagle,fire,fish,flower,fox,gun,heart,key,knife,leaf,lion,mermaid,octopus,owl,ribbon,rope,scorpion,shark,shield,skull,snake,spider,star,tiger`<br>fp:`anchor;bird;cat;crown;diamond;dog;fire;fish;flower;fox;gun;heart;key;knife;leaf;lion;mermaid;octopus;owl;ribbon;rope;scorpion;shark;shield;skull;snake;spider;star;tiger`<br>n_fp:`29` | `bird,eagle,fish,flower,lion,owl,shark,star,tiger,unknown,water`<br>fp:`bird;fish;flower;lion;owl;shark;star;tiger;unknown;water`<br>n_fp:`10` |
| `qwen2_5_vl` | `eagle,unknown`<br>fp:`unknown`<br>n_fp:`1` | `unknown`<br>fp:`unknown`<br>n_fp:`1` | `eagle,unknown`<br>fp:`unknown`<br>n_fp:`1` |

## image_id: `10257634316_82ecfe9f0f_z` (test_open)
**GT (image-level):** `eagle`

**Crop label:** `eagle.png`

| Model | Baseline (pred / fp / n_fp) | Crop black (pred / fp / n_fp) | Crop white (pred / fp / n_fp) |
|---|---|---|---|
| `gemma3` | `fire,snake,unknown`<br>fp:`fire;snake;unknown`<br>n_fp:`3` | `eagle;fire;water`<br>fp:`fire;water`<br>n_fp:`2` | `fire;snake`<br>fp:`fire;snake`<br>n_fp:`2` |
| `llama3_2_vision` | `bird,eagle,fire`<br>fp:`bird;fire`<br>n_fp:`2` | `bird,eagle,fish,flower,gun,key,leaf,lion,octopus,owl,shark,shield,skull,snake,spider,star,tiger,unknown,water,wolf`<br>fp:`bird;fish;flower;gun;key;leaf;lion;octopus;owl;shark;shield;skull;snake;spider;star;tiger;unknown;water;wolf`<br>n_fp:`19` | `bird,eagle,fish,flower,gun,key,leaf,lion,octopus,owl,star,tiger,unknown,water,wolf`<br>fp:`bird;fish;flower;gun;key;leaf;lion;octopus;owl;star;tiger;unknown;water;wolf`<br>n_fp:`14` |
| `qwen2_5_vl` | `bird,eagle,fire,unknown`<br>fp:`bird;fire;unknown`<br>n_fp:`3` | `eagle,unknown`<br>fp:`unknown`<br>n_fp:`1` | `eagle,unknown`<br>fp:`unknown`<br>n_fp:`1` |

## image_id: `196339401_64bbc02202_b` (test_open)
**GT (image-level):** `eagle;leaf;ribbon;star`

**Crop label:** `eagle.png`

| Model | Baseline (pred / fp / n_fp) | Crop black (pred / fp / n_fp) | Crop white (pred / fp / n_fp) |
|---|---|---|---|
| `gemma3` | `eagle,unknown,water`<br>fp:`unknown;water`<br>n_fp:`2` | `mermaid;octopus;water`<br>fp:`mermaid;octopus;water`<br>n_fp:`3` | `eagle;fire;water`<br>fp:`fire;water`<br>n_fp:`2` |
| `llama3_2_vision` | `eagle`<br>fp:``<br>n_fp:`0` | `eagle,unknown`<br>fp:`unknown`<br>n_fp:`1` | `eagle,unknown`<br>fp:`unknown`<br>n_fp:`1` |
| `qwen2_5_vl` | `eagle,ribbon,star`<br>fp:``<br>n_fp:`0` | `eagle`<br>fp:``<br>n_fp:`0` | `eagle`<br>fp:``<br>n_fp:`0` |
