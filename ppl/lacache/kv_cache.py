import math
import torch


class LaCache:
    def __init__(self, cache_size=256, span=16, overlap=0):
        self.cache_size = cache_size
        self.start_size = 4
        self.ladder_size = cache_size - 4
        self.span = span 
        self.overlap = overlap 
        self.ladder_cursor = self.start_size
        self.compensation_for_match_average_size = False


    def __call__(self, past_key_values): 
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(2)
        if seq_len <= self.cache_size:
            return past_key_values
        layer_number = len(past_key_values)

        ladder_kv_cache = []
        cache_size_per_ladder_before = 1 + math.ceil((layer_number - self.span)/(self.span - self.overlap))
        cache_size_per_ladder_after = min(math.ceil(self.span/(self.span-self.overlap)), cache_size_per_ladder_before)
        cache_size_per_ladder_can_reduce = cache_size_per_ladder_before - cache_size_per_ladder_after
        if not self.compensation_for_match_average_size:
            compensation = math.floor(cache_size_per_ladder_can_reduce/2)
            self.ladder_size += compensation
            self.cache_size += compensation
            self.compensation_for_match_average_size = True
        num_tokens_to_reduce = seq_len - self.cache_size
        cache_size_remaining_before_startover = self.cache_size - self.ladder_cursor - layer_number
        num_tokens_can_reduce_before_startover = (cache_size_remaining_before_startover // cache_size_per_ladder_after) * cache_size_per_ladder_can_reduce
        startover = (num_tokens_to_reduce > num_tokens_can_reduce_before_startover)
        if not startover:
            ladder_cursor_increment = math.ceil(num_tokens_to_reduce / cache_size_per_ladder_can_reduce) * cache_size_per_ladder_after
        else:
            ladder_cursor_increment = cache_size_remaining_before_startover
        ladder_number = ladder_cursor_increment // cache_size_per_ladder_after
        ladder_index_before_old_cursor = list(range(self.ladder_cursor))
        ladder_index_after_new_cursor = list(range(self.ladder_cursor + cache_size_per_ladder_before * ladder_number, seq_len))
        if startover:
            num_tokens_can_reduce_within_one_startover = ((self.cache_size - self.start_size - layer_number) // cache_size_per_ladder_after) * cache_size_per_ladder_can_reduce 
            num_tokens_to_reduce_remaining_after_startover = num_tokens_to_reduce - num_tokens_can_reduce_before_startover
            startover_times = math.ceil(num_tokens_to_reduce_remaining_after_startover / num_tokens_can_reduce_within_one_startover)
            cache_size_per_ladder_before_with_startover = cache_size_per_ladder_before + (startover_times - 1) * (cache_size_per_ladder_before - 1)
            cache_size_per_ladder_can_reduce_with_startover = cache_size_per_ladder_before_with_startover - cache_size_per_ladder_after
            ladder_cursor_increment_after_startover = math.ceil(num_tokens_to_reduce_remaining_after_startover / cache_size_per_ladder_can_reduce_with_startover) * cache_size_per_ladder_after
            ladder_number_with_startover = ladder_cursor_increment_after_startover // cache_size_per_ladder_after
        for i, (k, v) in enumerate(past_key_values):
            ladder_index_old_to_new_cursor = [self.ladder_cursor + i % (cache_size_per_ladder_can_reduce + 1) + \
                cache_size_per_ladder_before * j + k for j in range(ladder_number) for k in range(cache_size_per_ladder_after)] # stacked ladder

            ladder_index = ladder_index_before_old_cursor + ladder_index_old_to_new_cursor + ladder_index_after_new_cursor
            if startover:
                ladder_index_before_old_cursor_with_startover = list(range(self.start_size))
                ladder_index_old_to_new_cursor_with_startover = [
                        self.start_size + \
                        i % (cache_size_per_ladder_can_reduce_with_startover + 1) + \
                        cache_size_per_ladder_before_with_startover * j + \
                        startover_times * k for j in range(ladder_number_with_startover) for k in range(cache_size_per_ladder_after)
                    ]  
                ladder_index_after_new_cursor_with_startover = list(range(self.start_size + cache_size_per_ladder_before_with_startover * ladder_number_with_startover, len(ladder_index)))
                ladder_index_startover = ladder_index_before_old_cursor_with_startover + ladder_index_old_to_new_cursor_with_startover + ladder_index_after_new_cursor_with_startover
                ladder_index = torch.tensor(ladder_index)[ladder_index_startover]
                
            ladder_index = torch.as_tensor(ladder_index, dtype=torch.long, device=k.device).view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[3])
            ladder_k_cache, ladder_v_cache = torch.gather(k, 2, ladder_index), torch.gather(v, 2, ladder_index)
            ladder_kv_cache.append([ladder_k_cache, ladder_v_cache])

        if not startover:
            self.ladder_cursor += ladder_cursor_increment
        else:
            self.ladder_cursor = self.start_size + ladder_cursor_increment_after_startover 
        return ladder_kv_cache
