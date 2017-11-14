import random
import copy
import time
import sys
import itertools
import functools

height    = 16 # 8 #16 #16
width     = 30 #10 #16 #30
num_mines = 99 #10 #40 #99

# This is used to make the virtual solving perform much faster
dynamic = {}

# Print out the current state
def print_state(state):
  state_row = ""
  for cell in state:
    if cell == -1:
      state_row += "."
    else:
      state_row += str(cell)
    if len(state_row)%(width+1) == width:
      state_row += "\n"
  print state_row

# Print out the underlying mines
def print_board(board):
  board_row = ""
  for cell in board:
    if cell == 0:
      board_row += "."
    else:
      board_row += "X"
    if len(board_row)%(width+1) == width:
      board_row += "\n"
  print board_row

# Generate the layout of the mines. Either generates a
# totally random board (no arguments given) or a board
# where the first chosen square is a 0. 
def gen_board(pick=-1):
  disallowed_mines = get_neighbors(pick, False)
  a = range(width*height)
  for cell in disallowed_mines:
    a.remove(cell)
  mine_locations = random.sample(a, num_mines)
  board = [0]*width*height
  for i in mine_locations:
    board[i] = 1
  return board

# Returns neighbors of a picked cell. Can be inclusive
# or exclusive depending on the second argument
def get_neighbors(cell, exclude=True):
  neighbors = []
  if (cell < 0 or cell >= width*height):
    return neighbors
  r_edge, l_edge, t_edge, b_edge = -1,2,-1,2
  if cell%width == 0:
    r_edge = 0
  if cell%width == width-1:
    l_edge = 1
  if cell/width == 0:
    t_edge = 0
  if cell/width == height-1:
    b_edge = 1
  for i in range(r_edge, l_edge):
    for j in range(t_edge, b_edge):
      neighbors.append(cell + i + width*j)
  if (exclude):
    neighbors.remove(cell)
  return neighbors

# Returns -1 on a loss, 1 on a win, 0 otherwise
def select(cells, _board, state):
  for cell in cells:
    if (cell < 0 or cell >= width*height):
      continue
    # You Lost :(
    if _board[cell] == 1:
      return -1
    mines_around = 0
    for i in get_neighbors(cell):
      if _board[i] == 1:
        mines_around += 1
    state[cell] = mines_around
  unknowns = 0
  # Check for win (all non-mines have been opened)
  for cell in state:
    if cell == -1:
      unknowns += 1
  if unknowns == sum(_board):
    return 1
  # You're still in it
  return 0

# Compose a list of all our information (cells with numbers)
# Separate them into edges (this is important when the remaining mines are low)
# Return list of edges, as well as list of non-edge cells
def get_edges(state):
  knowns = set([])
  info = []
  for cell in range(height*width):
    if state[cell] != -1:
      knowns.add(cell)
      neighbors = get_neighbors(cell)
      knowns.update(neighbors)
      for neighbor in neighbors[:]:
        if state[neighbor] != -1:
          neighbors.remove(neighbor)
      if len(neighbors) > 0:
        info.append([cell, state[cell], neighbors])
  non_edges = list(set(range(height*width)) - knowns)
  # Split up info into different connected borders
  # This is a graph connectivity problem, so there might be a nicer solution
  # But there is a lot of overlap for the graph edges, so it shouldn't
  # take more than two or three passes through the loop
  edges = []
  edge_cells = [] # I'm using this to make my life much easier
  while len(info) > 0:
    edge_cells.append(info[0][2])
    edges.append([info[0]])
    info.remove(info[0])
    more_cells = True
    while more_cells:
      more_cells = False
      for cell, number, neighbors in info[:]:
        add_this_on = False
        for neighbor in neighbors:
          if neighbor in edge_cells[len(edges)-1]:
            more_cells = True
            add_this_on = True
        if add_this_on:
          edge_cells[-1] = list(set(edge_cells[-1] + neighbors))
          edges[-1].append([cell, number, neighbors])
          info.remove([cell, number, neighbors])
  return edges, non_edges, edge_cells

def get_atoms(state, edge_cells):
  atoms = []
  for edge in edge_cells:
    atoms_map = []
    for edge_cell in edge:
      touching = []
      for neighbor in get_neighbors(edge_cell):
        if state[neighbor] != -1:
          touching.append(neighbor)
      atoms_map.append([touching, edge_cell])
    atoms_map.sort()
    current = atoms_map[0][0]
    edge_atoms = [[]]
    for touching, edge_cell in atoms_map:
      if touching == current:
        edge_atoms[-1].append(edge_cell)
      else:
        current = touching
        edge_atoms.append([edge_cell])
    atoms += edge_atoms
  atom_dict = {}
  for atom in atoms:
    for cell in atom:
      atom_dict[cell] = atom
  return atom_dict

def repurpose_edges(edges, atom_dict):
  for edge in edges:
    for i in range(len(edge)):
      repurposed_neighbors = []
      for neighbor in edge[i][2]:
        if atom_dict[neighbor] not in repurposed_neighbors:
          repurposed_neighbors.append(atom_dict[neighbor])
      edge[i][2] = repurposed_neighbors

def expand(neighbors):
  if len(neighbors) == 0:
    return []
  return reduce(lambda x,y: x+y, neighbors)

def expand_edge(edge, atom_dict, cutoff):
  expanded_edge = []
  all_mines = set(edge[0][0]+edge[0][1])
  #sum_ways = 0
  for mines, not_mines, ways in edge[:]:
    if ways > cutoff:
      continue
    seen_mines = set([])
    #sum_ways += ways #for debugging. can check this against len(expanded_edge)
    expand = [[]]
    for mine in mines:
      if mine in seen_mines:
        continue
      atom = atom_dict[mine]
      seen_mines.update(atom)
      newlist = []
      while len(expand) > 0:
        mine_arrangement = expand.pop()
        extras = list(itertools.combinations(atom, len(set(atom)&set(mines))))
        for extra in extras:
          newlist.append(mine_arrangement+list(extra))
      expand += newlist
    for i in expand:
      expanded_edge.append([i, list(all_mines - set(i))])
  return expanded_edge

# This returns all non-mines that can be immediately deduced
def get_obvious_mines(_edge):
  edge = copy.deepcopy(_edge)
  mines = []
  not_mines = []
  deducing = True
  while deducing:
    deducing = False
    for cell, number, neighbors in edge[:]:
      if number == len(neighbors):
	deducing = True
	mines = list(set(mines + neighbors))
	edge.remove([cell, number, neighbors])
      elif number == 0: # This must be elif to ensure we don't try to remove it twice when we have [x, 0, []]
	deducing = True
	not_mines = list(set(not_mines + neighbors))
	edge.remove([cell, number, neighbors])
    for i, [cell, number, neighbors] in enumerate(edge):
      for mine in mines:
	if mine in neighbors:
	  deducing = True
	  neighbors.remove(mine)
	  edge[i][1] = edge[i][1] - 1
      for not_mine in not_mines:
	if not_mine in neighbors:
	  deducing = True
	  neighbors.remove(not_mine)
  return not_mines

# This method returns all arrangements of mines touching numbers we've opened
# Recursively places mines, along border and checks for contradictions.
# Returns 0 if a contradiction is reached, otherwise returns list of mines and non-mines
def get_mines(_edge, _mines, _not_mines, atom_dict):
  edge = copy.deepcopy(_edge)
  mines = _mines[:]
  not_mines = _not_mines[:]
  deducing = True
  while deducing:
    deducing = False
    for cell, number, long_neighbors in edge[:]:
      neighbors = expand(long_neighbors)
      if number < 0 or number > len(neighbors):
        # We have reached a contradiction
        return 0
      if number == len(neighbors):
        deducing = True
        mines = list(set(mines + neighbors))
        edge.remove([cell, number, long_neighbors])
      elif number == 0: # This must be elif to ensure we don't try to remove it twice when we have [x, 0, []]
        deducing = True
        not_mines = list(set(not_mines + neighbors))
        edge.remove([cell, number, long_neighbors])
    for i, [cell, number, long_neighbors] in enumerate(edge):
      neighbors = expand(long_neighbors)
      for mine in mines:
        if atom_dict[mine] in long_neighbors:
          deducing = True
          long_neighbors.remove(atom_dict[mine])
          edge[i][1] -= len(set(atom_dict[mine]) & set(mines))
      for not_mine in not_mines:
        if atom_dict[not_mine] in long_neighbors:
          deducing = True
          long_neighbors.remove(atom_dict[not_mine])
    if len(set(mines) & set(not_mines)) > 0:
      # Contradiction as the sets should be exclusive
      return 0

  # We have completed the border
  if len(edge) == 0:
    return [[mines, not_mines, 1]]
 
  # Now perform trial and error to find next obvious mines
  # Maintain a list of possible mine arrangements for each edge
  # Splitting up the independent edges keeps these lists small
  # If you have 3 edges with a,b,c possibilities we only
  # need to keep a+b+c not a*b*c, as each edge is independent
  arrangements = []
  already_tried = []
  # Originally, this program was written with each cell as the smallest unit
  # But this could lead to scenarios where the trial and error method would
  # blow up exponentially in trying to enumerate all possible arrangements.
  # This forced me to adapt the program. I noticed that certain cells just
  # off the edge could be grouped together, and be thought of logically as one
  # entity, which I called an atom. Then a cell could have for example 2 mines
  # in a 4 cell atom, and I would only need to record one of the arrangements
  # as long I multiplied by the total number 6 in the end. This rescued my 
  # program from being susceptible to extremely high numbers of arrangements,
  # although the repurposing has made the code harder to read.
  #
  # For each atom, try putting in as many mines as it will take
  number = edge[0][1]
  for i in range(len(edge[0][2])):
    trial_atom = edge[0][2].pop()
    for j in range(min(len(trial_atom), number), 0, -1):
      edge[0][1] = number - j
      x = get_mines(edge, mines+trial_atom[:j], not_mines+trial_atom[j:]+already_tried, atom_dict)
      if x != 0:
        mult = choose(len(trial_atom), j, 100)
        x = map(lambda y:[y[0], y[1], mult*y[2]], x)
        arrangements += x
    already_tried += trial_atom

  return arrangements

# Given a list of arrangements for every edge, group them all together
# by how many total mines they use. This will be used to calculate total
# number of boards remaining, as well as probabilities that certain cells
# are mines or not
# This is essentially a map-reduce method.
def handle_arrangements(arrangements):
  handled = [[0,1]]	# [[number of mines, ways it can be made]]
  for edge in arrangements:
    temp = []
    for mines, not_mines, ways in edge:
      temp += map(lambda (x,y):[x+len(mines),y*ways], handled)
    temp.sort(key=lambda (x,y):x)
    # Merge together arrangements that use same number of mines
    handled = []
    current = 0
    for mines, count in temp:
      if mines == current:
        handled[-1][1] += count
      else:
        handled.append([mines, count])
        current = mines
  return handled

# This is very very similar to the above method, except this time keep
# track of how each total mines was achieved (i.e. 14 = 6 + 8 vs 7 + 7)
# This will be needed in order to generate random boards from a given state
# When the above method was written, I didn't know this one would be needed.
# The above method is redundant, but the solution is too clean for me to
# scrap the code without dying a little on the inside.
def extra_handle(arrangements):
  handled = [[[], 1]]
  for edge in arrangements:
    condensed_edge = []
    temp = map(lambda (x,y,z):[len(x),z], edge)
    temp.sort()
    current = 0
    for mines, count in temp:
      if mines == current:
        condensed_edge[-1][1] += count
      else:
        condensed_edge.append([mines, count])
        current = mines
    temp = []
    for mines, count in condensed_edge:
      temp += map(lambda (x,y):[x+[mines],y*count], handled)
    handled = temp
  distributions = {}
  for ordering, ways in handled:
    total = sum(ordering)
    if total in distributions:
      distributions[total].append([ordering, ways])
    else:
      distributions[total] = [[ordering, ways]]
  return distributions

# Returns n choose k, as long as it is less than the cutoff
# Otherwise, returns -1. 
def choose(n, k, cutoff):
  if (k > n/2):
    k = n - k
  if k < 0:
    return 0
  combinations = 1
  for i in range(k):
    combinations *= n-i
    combinations /= i+1
    if combinations > cutoff:
      return -1
  return combinations

# Returns number of boards with a given edge arrangement.
# If the number is higher than our cutoff, return -1    
def get_num_boards(handled, non_edges, total_mines, cutoff):
  total = 0
  for [mines, count] in handled:
    combinations = choose(non_edges, total_mines - mines, cutoff)
    if combinations == -1:
      return -1
    total += combinations*count
  if total > cutoff:
    return -1
  return total

# Generate all possible boards for the given arrangements
# Returns boards and a list of cells that should NOT be selected
def gen_virtual_boards(exp_arrangements, non_edges, total_mines):
  mines_list = [[]]
  # First trim the edges to be as small as possible. This isn't 
  # necessary but it makes the output look much cleaner.
  unknowns = set([])
  definite_mines = set([])
  for edge in exp_arrangements:
    new_definite_mines = set(edge[0][0])
    for [mines, not_mines] in edge:
      unknowns.update(not_mines)
      new_definite_mines &= set(mines)
    definite_mines.update(new_definite_mines)
  necessary_mines = set([])
  for cell in list(unknowns)+non_edges:
    necessary_mines.update(get_neighbors(cell))
  necessary_mines &= definite_mines
  # If there is a definite mine in an arrangement, which doesn't
  # border an unknown cell, there is no need to keep track of it.
  b = necessary_mines | unknowns
  for edge in exp_arrangements:
    mines = map(lambda (x,y):list(set(x)&b), edge)
    mines_list = [x+y for x in mines for y in mines_list]
  t_mines = total_mines - len(definite_mines) + len(necessary_mines)
  # Now generate the boards with the given mine placements
  # Store the possibilities for the non_edges dynamically to save a bit of time
  non_edge_patterns = {}
  boards = []
  for mines in mines_list:
    if len(mines) > t_mines or len(mines) + len(non_edges) < t_mines:
      continue
    difference = t_mines - len(mines)
    if difference not in non_edge_patterns:
      temp = list(itertools.combinations(non_edges, difference))
      non_edge_patterns[difference] = [list(i) for i in temp]
    for pattern in non_edge_patterns[difference]:
      all_mines = mines + pattern
      vboard = [0]*height*width
      for mine in all_mines:
        vboard[mine] = 1
      boards.append([vboard, len(all_mines)])
  #for vboard, vmines in boards:
  #  print_board(vboard)
  return boards, list(b)+non_edges

def virtual_solve(exp_arrangements, non_edges, total_mines):
  boards, options = gen_virtual_boards(exp_arrangements, non_edges, total_mines)
  to_select = list(set(range(height*width)) - set(options))
  #state = [-1]*height*width
  #select(to_select, boards[0][0], state)
  
  # Keep track of minimum losses, so you can cut it off if it goes above
  # that number
  min_losses = len(boards)
  best_cell = 0
  for cell in options:
    losses = 0
    for vboard, vmines in boards:
      #temp_state = state[:]
      temp_state = [-1]*height*width
      select(to_select, vboard, temp_state)
      still_in = select([cell], vboard, temp_state)
      while still_in == 0:
        selection = make_selection(temp_state, vmines, vboard)
        #selection = make_selection(temp_state, vmines)
        still_in = select(selection, vboard, temp_state)
      if still_in == -1:
        losses += 1
      if losses >= min_losses:
        break
    if losses < min_losses:
      min_losses = losses
      best_cell = cell
  #return best_cell
  return best_cell, 1.0 - 1.0*losses/len(boards)

# This takes a lot of inputs and returns a list of possible boards that were
# selected randomly from the current state.
#
# The problem of generating a uniformly random board was a very tricky one.
# In the end I settled on first picking the # of mines that would be
# contained in the edge_cells. Next I pick the # of mines for each edge that
# will add up to the desired total. Lastly for each I pick the arrangement
# for the given number of mines. As long as I am making each choice with a
# probability that reflects how often it shows up, the distribution will end
# up being uniform.
def gen_virtual_board_mc(number, arrangements, non_edges, total_mines, handled, extra_handled, atom_dict):
  total_probs = []
  ne = len(non_edges)
  numerator = 0.0
  denominator = 0.0
  remain0 = total_mines - handled[0][0]
  for [mines, ways] in handled:
    remaining = total_mines - mines
    # Instead of computing massive combinations, I have simplified the work the computer has to do
    # by factoring out an (ne CHOOSE remain0) from the numerator and denominator.
    factor = 1.0
    for i in range(remain0, remaining, -1):
      factor *= 1.0*i / (ne - i + 1)
    numerator += ways * factor
    denominator += ways * factor
    total_probs.append(numerator)
  
  # Preprocess the edges
  edge_bounds = {}
  for i, edge in enumerate(arrangements):
    for mines, not_mines, ways in edge:
      if (i, len(mines)) in edge_bounds:
        edge_bounds[(i, len(mines))] += ways
      else:
        edge_bounds[(i, len(mines))] = ways

  new_boards = []
  total_bounds = {}
  for i in range(number):
    # Randomly select how many total mines will be on the edges
    chosen_mine_total = handled[-1][0]
    gen = random.random()*denominator
    for j, prob in enumerate(total_probs):
      if prob >= gen: # This could be done slightly faster with a binary search
        chosen_mine_total = handled[j][0]
        break
    # Now randomly select the # of mines for each side given the chosen total
    edge_options = extra_handled[chosen_mine_total]
    if chosen_mine_total not in total_bounds:
      total_bounds[chosen_mine_total] = sum([ways for arr,ways in edge_options])
    upper_bound = total_bounds[chosen_mine_total]
    gen = random.randint(1, upper_bound)
    chosen_edge_option = edge_options[-1][0]
    for j, [arr, ways] in enumerate(edge_options):
      gen -= ways
      if gen <= 0:
        chosen_edge_option = edge_options[j][0]
        break
    # Now for each edge, randomly select an arrangement with the desired
    # number of mines
    new_mines = []
    for j, len_mines in enumerate(chosen_edge_option):
      gen = random.randint(1, edge_bounds[(j, len_mines)])
      for k, [mines, not_mines, ways] in enumerate(arrangements[j]):
        if len(mines) == len_mines:
          gen -= ways
          if gen <= 0:
            seen = set([])
            for cell in mines:
              # Lastly, given an arrangement, randomly select which cells in an
              # atom will be mines
              if cell not in seen:
                atom = atom_dict[cell]
                count = len(set(atom)&set(mines))
                new_mines += random.sample(atom, count)
                seen.update(atom)
            break
    # Add on the mines that come from the non-edge cells
    new_mines += random.sample(non_edges, total_mines - len(new_mines))
    new_board = [0]*height*width
    for cell in new_mines:
      new_board[cell] = 1
    #print_board(new_board)
    new_boards.append(new_board)
  return new_boards

# This method is not currently in use. Initially I thought I would be able to 
# simulate some number of picks down the line any time there is nothing more
# to be logically deduced. It turns out that even going one pick deep takes
# a lot of time. This method could become viable if I implement an update
# method that simply updates existing variables instead of recalculating them
# from scratch after each selection.
def virtual_solve_mc(vboards, total_mines, options, safe_picks, depth):
  #state = [-1]*height*width
  #select(safe_picks, vboards[0], state)
  losses = []
  # Keep track of minimum losses, so you can cut it off if it goes above
  # that number
  min_losses = len(vboards)
  best_cell = 0
  for cell in options:
    cur_losses = 0
    for vboard in vboards:
      #temp_state = state[:]
      temp_state = [-1]*height*width
      select(safe_picks, vboard, temp_state)
      still_in = select([cell], vboard, temp_state)
      while still_in == 0:
        selection = make_selection(temp_state, sum(vboard), vboard[:], depth)
        still_in = select(selection, vboard, temp_state)
      if still_in == -1:
        cur_losses += 1
      # Can edit this to cut off at a certain point like 2x min_losses
      if cur_losses > 2*min_losses:
        losses.append(cur_losses)
        break
    if cur_losses < min_losses:
      min_losses = cur_losses
      best_cell = cell
    losses.append(cur_losses)
  return losses, best_cell

# The data looked like it would lend itself to a cubic approximation
def cubic_interpolation(x):
  if x > .31:
    return 0
  if x < .115:
    return 1
  return 92.25*x*x*x - 56.905*x*x + 5.798*x + .9315

# Return a cell with the best heuristic pick.
# Compute the heuristic by generating random boards, and for each board
# Keep track of how many times it is solved. If we reach a point where
# we must make a guess, then approximate our chance of solving it as a 
# function of ~ the number of mines / the remaining edge cells.
#
# The approximation could also take into account the length of the edges
# and the shape of the non-edges. At the moment they are not used.
def h_pick(vboards, total_mines, options, safe_picks, depth, ps):
  # The value of ratio can be toyed with. It represents the ratio of
  # our guessed probability to solve the puzzle to the total % of the puzzle
  # solved, which add together to get our heuristic.
  ratio = .5
  scores = []
  best_score = 0
  best_cell = options[0]
  for cell in options:
    cur_score = 0
    for vboard in vboards:
      temp_state = [-1]*height*width
      select(safe_picks, vboard, temp_state)
      still_in = select([cell], vboard, temp_state)
      while still_in == 0:
        selection = make_selection(temp_state, sum(vboard), vboard, depth)
        if selection[0] == -1:
          for i in temp_state:
            if i != -1:
              cur_score += ratio / (height*width - total_mines)
          ne = len(selection[1])
          prob = selection[2]
          y = cubic_interpolation(prob)
          # The smaller the number of non-edges, the more closely the probability
          # of solving will resemble the ratio of mines to remaining edge cells
          # So, if we are within some cutoff return a score that is one part the 
          # probability, and one part the function of the probability
          cutoff = 10.0
          if ne > cutoff:
            cur_score += y
          else:
            cur_score += (1.0 - prob)*(cutoff - ne)/cutoff + y*ne/cutoff
          break
        still_in = select(selection, vboard, temp_state)
      if still_in == 1:
        cur_score += 1.0 + ratio
    scores.append(cur_score)
    if cur_score > best_score:
      best_score = cur_score
      best_cell = cell
  # This is still a work in progress, so these print calls are to help me
  # see inside the solving process.
  print "cell, score, prob. YOUR WINNER: ", best_cell
  for i, cell in enumerate(options):
    print cell, "\t", scores[i], "\t", 10.0-10*ps[cell]
  return best_cell

# Given a list of cells return a subset of this list that are AT BEST the worst
# picks possible. For a given cell this means that if in fact it is not a mine
# you will not gain any additional information from opening it.
def get_unoptimal(cells, arrangements, edge_cells, non_edges, atom_dict):
  unoptimal = []
  for cell in cells:
    neighbors = get_neighbors(cell)
    if len(set(neighbors)&set(non_edges)) > 0:
      continue
    n_atoms = []
    if cell in non_edges:
      n_atoms.append([cell])
    else:
      n_atoms.append(atom_dict[cell])
    for neighbor in neighbors:
      if neighbor in edge_cells:
        atom = atom_dict[neighbor]
        if atom not in n_atoms:
          n_atoms.append(atom)
    cell_atom = n_atoms.pop(0)
    mismatch = False
    for edge in arrangements:
      if mismatch:
        continue
      prev_count = -1
      for [mines, not_mines, ways] in edge:
        count = len(set(cell_atom)&set(mines))
        # Skip over arrangements where the cell in question is a mine
        if cell in mines and len(cell_atom) == count:
          continue
        # Because the edges are condensed into atoms, we need to ensure that 
        # the arrangements that haven't been enumerated are taken into consideration
        # This means that if a cell in an atom, not contained in neighbors can be
        # either a mine or a non-mine, the cell won't be unoptimal.
        if len(set(cell_atom)-set(neighbors)) > 1:
          if count != 0 and len(cell_atom) - count > 1:
            mismatch = True
            continue
        for atom in n_atoms:
          a_count = len(set(atom)&set(mines))
          if len(set(atom)-set(neighbors)) > 0:
            if a_count != 0 and len(atom) - a_count > 0:
              mismatch = True
          count += a_count
        if prev_count < 0:
          prev_count = count
        if count != prev_count:
          mismatch = True
    if not mismatch:
      unoptimal.append(cell)
  return unoptimal

# Given an edge, return a minimal list of cells that should be picked
# It is important that this list is small
# If A is not a mine implies B is not a mine, but the reverse isn't true,
# eliminate A from the list. If A->B and B->A, then just pick one of them.
def get_dependencies(edge, atom_dict):
  edge_cells = edge[0][0]+edge[0][1]
  dependencies = {}
  for cell in edge_cells:
    dependencies[cell] = set(edge_cells)
    for [mines, non_mines, ways] in edge:
      atom = atom_dict[cell]
      if len(set(atom)-set(mines)) == 0:
        continue
      mine_dep = set([])
      for mine in mines:
        mine_dep.update(atom_dict[mine])
      mine_dep -= set([cell])
      dependencies[cell] -= mine_dep
  independent = []
  dependent = set([])
  for cell in edge_cells:
    dependencies[cell] -= dependent
    if len(dependencies[cell]) == 1:
      independent.append(cell)
    else:
      dependent.add(cell)
  return independent
 
# Return the probability that a given non-edge cell is a mine
def get_probability_non_edge(cell, handled, non_edges):
  # p = number of boards where cell is a mine / total number of boards
  # In the case of a non-edge, we can sum over all arangements with a given number of mines
  # so p = sum Pr[cell is mine | x total mines]*Pr[x total mines]
  ne = len(non_edges)
  numerator = 0.0
  denominator = 0.0
  remain0 = num_mines - handled[0][0]
  for [mines, ways] in handled:
    remaining = num_mines - mines
    # Instead of computing massive combinations, I have simplified the work the computer has to do
    # by factoring out an (ne CHOOSE remain0) from the numerator and denominator.
    factor = 1.0
    for i in range(remain0, remaining, -1):
      factor *= 1.0*i / (ne - i + 1)
    prob = 1.0*remaining / ne
    numerator += ways * factor * prob
    denominator += ways * factor
  return numerator / denominator

# Return the probability that a given edge cell is a mine
def get_probability_edge(cell, handled, non_edges, edge, atom_dict):
  ne = len(non_edges)
  numerator = 0.0
  denominator = 0.0
  remain0 = num_mines - handled[0][0] - len(edge[0][0])
  for [mines, ways] in handled:
    for [a_mines, a_non_mines, a_ways] in edge:
      remaining = num_mines - mines - len(a_mines)
      # Instead of computing massive combinations, I have simplified the work the computer has to do
      # by factoring out an (ne CHOOSE remain0) from the numerator and denominator.
      factor = 1.0
      for i in range(remain0, remaining, -1):
        factor *= 1.0*i / (ne - i + 1)
      # In the case of a cell on the edge, we do the same thing, but determine
      # Pr[cell is mine | x total mines] by counting how many arrangements have the cell as a non-mine
      atom = atom_dict[cell]
      prob = 1.0 * len(set(a_mines)&set(atom)) / len(atom)
      numerator += ways * a_ways * factor * prob
      denominator += ways * a_ways * factor
  return numerator / denominator

# This method separates the non_edge cells into independent regions
# It then returns the cell from the largest region which has the highest chance
# of being an effective 0. 
def non_edge_pick(state, edges, non_edges):
  seen_non_edges = set([])
  regions = []
  for cell in non_edges:
    if cell in seen_non_edges:
      continue
    current_region = set([])
    doing_now = set([cell])
    while len(doing_now) > 0:
      seen_non_edges.update(doing_now)
      current_region.update(doing_now)
      cur_cell = doing_now.pop()
      doing_now.update(set(get_neighbors(cur_cell))&set(non_edges) - seen_non_edges)
    regions.append(list(current_region))
  regions.sort(key=lambda x:len(x))

  edge_dict = {}
  for edge in edges:
    for cell, number, neighbors in edge:
      edge_dict[cell] = set(expand(neighbors))

  interior_edges = {}
  for cell in non_edges:
    neighbors = set(get_neighbors(cell))
    known_neighbors = set(neighbors)-set(non_edges)
    if len(known_neighbors) != 0 or (1 + cell)%width < 2 or cell/width == 0 or cell/width == height - 1:      
      second_cousins = []
      for i in range(-1,2):
        second_cousins += [cell-2 + i*width, cell+2 + i*width, cell+i - 2*width, cell+i + 2*width]
      for nbor in second_cousins:
        if nbor in edge_dict:
          if edge_dict[nbor] <= neighbors:
            neighbors -= edge_dict[nbor]
      directions = [x - cell for x in neighbors]
      flag = True
      distance = 0
      while (flag):
        distance += 1
        rays = [distance*x + cell for x in directions]
        for ray in rays:
          if ray < 0 or ray >= height*width:
            flag = False
          elif ray not in non_edges:
            flag = False
      interior_edges[cell] = [len(neighbors), distance]

  min_neighbors = 8
  min_cell = non_edges[0]
  max_dist = min(height, width)
  for cell in interior_edges:
    num_nbor, dist = interior_edges[cell]
    if num_nbor < min_neighbors:
      min_neighbors = num_nbor
      min_cell = cell
      max_dist = dist
    if num_nbor == min_neighbors and dist > max_dist:
      min_neighbors = num_nbor
      min_cell = cell
      max_dist = dist
  return min_cell

# This is essentially the entire program
def make_selection(state, total_mines, tboard=[], depth=1):
  if (tuple(state), total_mines) in dynamic:
    return dynamic[(tuple(state), total_mines)]
  edges, non_edges, edge_cells = get_edges(state)
  all_safe = []
  for edge in edges:
    all_safe = list(set(all_safe + get_obvious_mines(edge)))
  # Don't resort to trial and error unless we must
  # This is done purely for speed purposes
  if len(all_safe) > 0:
    return all_safe

  atom_dict = get_atoms(state, edge_cells)
  repurpose_edges(edges, atom_dict)

  min_mines = 0
  max_mines = 0
  arrangements = []
  for edge in edges:
    safe_picks = []
    edge_arrangements = get_mines(edge, [], [], atom_dict)
    # TODO something breaks here in the virtual_solve_mc method. I'm assuming it has to do with edges that touch each other with no non-edges
    try:
      test = edge_arrangements[0]
    except:
      print_state(state)
      print_board(tboard)
      print "EDGE_ARR: ", edge_arrangements
      print "EDGE: ", edge
    edge_arrangements = sorted(edge_arrangements, key=lambda x: len(x[0]))

    for j, [mines, not_mines, ways] in enumerate(edge_arrangements):
      if j == 0:
        safe_picks = not_mines[:]
      else:
        safe_picks = list(set(not_mines) & set(safe_picks))
    all_safe = list(set(all_safe + safe_picks))
    if len(edge_arrangements) > 0:
      arrangements.append(edge_arrangements)
      min_mines += len(edge_arrangements[0][0])
      max_mines += len(edge_arrangements[-1][0])

  # More repurposing is needed here. I need to make sure cells in an atom
  # are ALL safe, because I am only keeping one atomic arrangement of 
  # mines and non-mines
  unsafe = set([])
  for i, cell in enumerate(all_safe):
    for atom_cell in atom_dict[cell]:
      if atom_cell not in all_safe:
        unsafe.add(cell)
  all_safe = list(set(all_safe) - unsafe)

  if len(all_safe) > 0:
    return all_safe
 
  # If an arrangement requires more mines than the total number in the board,
  # Or if it contains too few mines to reach the total number of mines in the board,
  # Then eliminate that arrangement, and check again for safe picks.
  recheck = False
  for edge in arrangements:
    old_edge = edge[:]
    while (min_mines - len(edge[0][0]) + len(edge[-1][0]) > total_mines):
      max_mines += -len(edge[-1][0]) + len(edge[-2][0])
      edge.pop()
      recheck = True
    while (max_mines - len(edge[-1][0]) + len(edge[0][0]) + len(non_edges) < total_mines):
      edge.pop(0)
      recheck = True
  if (recheck):
    for edge in arrangements:
      safe_picks = []
      for j, [mines, not_mines, ways] in enumerate(edge):
        if j == 0:
          safe_picks = not_mines[:]
        else:
          safe_picks = list(set(not_mines)&set(safe_picks))
      all_safe = list(set(all_safe + safe_picks))
  if len(all_safe) > 0:
    return all_safe  

  # Next calculate how many valid boards are still possible.
  # If the number is small enough employ a brute force solution
  # that is guaranteed to be optimal
  handled = handle_arrangements(arrangements)
  extra_handled = extra_handle(arrangements)
  while handled[-1][0] > total_mines:
    extra_handled.pop(handled.pop()[0])

  while handled[0][0] + len(non_edges) < total_mines:
    extra_handled.pop(handled.pop(0)[0])

  # If all the mines are on the edges, open up all non edge cells
  if handled[0][0] == total_mines and len(non_edges) > 0:
    return non_edges

  # These lines have to do with the h_pick function. If we're in the MC
  # method, then don't bother wasting time using virtual solve.
  # Just return an escape sequence, with pertinent information for the fn.
  if depth > 1:
    if len(non_edges) == 0:
      return [-1, [], .5, atom_dict]
    approx_prob = get_probability_non_edge(non_edges[0], handled, non_edges)
    return [-1, non_edges, approx_prob, atom_dict]

  cutoff = 50
  exp_arrangements = [expand_edge(i, atom_dict, 2*cutoff) for i in arrangements]

  # Optimally solve any edges for which no new information can be learned
  # These are the edges that don't touch any non-edge cells

  # TODO 
  # merge edges together that touch. if this is not done program might break
  for edge in exp_arrangements:
    if len(edge) == 0:
      continue
    closed = True
    trivial = False
    if len(edge[0][1]) == 0:
      trivial = True
    else:
      for cell in edge[0][0]+edge[0][1]:
        if len(set(non_edges) & set(get_neighbors(cell))) > 0:
          closed = False
      # Also make sure that all arrangements require the same number of mines
      for mines, not_mines in edge:
        if len(mines) != len(edge[0][0]):
          closed = False
    if closed and not trivial:
      #print "VIRTUAL INSANITY"
      a,b = virtual_solve([edge], [], len(edge[0][0]))
      dynamic[(tuple(state), total_mines)] = [a]
      return [a]

  # With no cutoff, this would find an optimal strategy for solving the entire
  # puzzle. This is computationally impossible though, which is what makes
  # minesweeper an interesting game.
  cutoff = 50
  num_boards = get_num_boards(handled, len(non_edges), total_mines, cutoff)
  if num_boards != -1:
    a,b = virtual_solve(exp_arrangements, non_edges, total_mines)
    dynamic[(tuple(state), total_mines)] = [a]
    return [a]

  # Now if there are any improvements to be made on this function, in terms
  # of selection accuracy, it is here.
  
  #print_state(state)
  options = []
  for i, edge in enumerate(arrangements):
    options += get_dependencies(edge, atom_dict)
  nep = non_edge_pick(state, edges, non_edges)
  #options.append(nep)

  ps = {}
  for i, edge in enumerate(arrangements):
    handled_others = handle_arrangements(arrangements[:i]+arrangements[i+1:])
    for cell in edge[0][0]+edge[0][1]:
      ps[cell] = get_probability_edge(cell, handled_others, non_edges, edge, atom_dict)
  ps[nep] = get_probability_non_edge(nep, handled, non_edges)

  # TODO
  # investigate whether this actually helps. i think it will
  #unoptimal = get_unoptimal(_CELLS TO TEST_, arrangements, expand(edge_cells), non_edges, atom_dict)
  
  # TODO annotate this better. trying out the modified MC heuristic method
  vboards = gen_virtual_board_mc(10, arrangements, non_edges, total_mines, handled, extra_handled, atom_dict)
  known_cells = []
  for i in range(height*width):
    if state[i] != -1:
      known_cells.append(i)
  reduced_options = sorted(options, key=lambda x:ps[x])
  reduced_options = reduced_options[:2]
  reduced_options.append(nep)

  hpick = h_pick(vboards, total_mines, reduced_options, known_cells, depth+1, ps)
  return [hpick]

  if depth < 1:
    known_cells = []
    for i in range(height*width):
      if state[i] != -1:
        known_cells.append(i)
    vboards = gen_virtual_board_mc(10, arrangements, non_edges, total_mines, handled, extra_handled, atom_dict)

    mc_losses, best_cell = virtual_solve_mc(vboards, total_mines, options, known_cells, depth+1)
    return [best_cell]
  else:
    best_cell = 0
    best_prob = 1
    for cell in ps:
      if ps[cell] < best_prob:
        best_cell = cell
        best_prob = ps[cell]
    
    if ps[nep] < best_prob:
      return [nep]
    else:
      return [best_cell]

  return [non_edge_pick(state, arrangements, non_edges)]


# Define state. -1: unknown. #: neighboring mines.
draw_count = 0
win_count = 0
t_start = time.time()
for i in range(100):
  state = [-1]*height*width
  board = gen_board(15*17)
  select([15*17], board, state)
  still_in = 0
  going_on = True
  while (still_in == 0 and going_on):
    #print_state(state)
    selection = make_selection(state, num_mines)
    still_in = select(selection, board, state)
    if still_in == 1:
      win_count = win_count + 1
    if len(selection) == 0:
      print_state(state)
      going_on = False
  print_state(state)
  print "\n============\n"
  print win_count, "/", i+1-draw_count
  print "\n============\n"
print "TIME: ", time.time() - t_start





