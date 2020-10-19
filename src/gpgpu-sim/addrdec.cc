// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt, Ali Bakhoda,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "addrdec.h"
#include <math.h>
#include <string.h>
#include "../option_parser.h"
#include "gpu-sim.h"
#include "hashing.h"

static long int powli(long int x, long int y);
static unsigned int LOGB2_32(unsigned int v);
static unsigned next_powerOf2(unsigned n);

static new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                      unsigned char high, unsigned char low);
static void addrdec_getmasklimit(new_addr_type mask, unsigned char *high,
                                 unsigned char *low);

linear_to_raw_address_translation::linear_to_raw_address_translation() {
  addrdec_option = NULL;
  ADDR_CHIP_S = 10;
  memset(addrdec_mklow, 0, N_ADDRDEC);
  memset(addrdec_mkhigh, 64, N_ADDRDEC);
  addrdec_mask[0] = 0x0000000000001C00;
  addrdec_mask[1] = 0x0000000000000300;
  addrdec_mask[2] = 0x000000000FFF0000;
  addrdec_mask[3] = 0x000000000000E0FF;
  addrdec_mask[4] = 0x000000000000000F;
}

void linear_to_raw_address_translation::addrdec_setoption(option_parser_t opp) {
  option_parser_register(opp, "-gpgpu_mem_addr_mapping", OPT_CSTR,
                         &addrdec_option,
                         "mapping memory address to dram model {dramid@<start "
                         "bit>;<memory address map>}",
                         NULL);
  option_parser_register(
      opp, "-gpgpu_mem_addr_test", OPT_BOOL, &run_test,
      "run sweep test to check address mapping for aliased address", "0");
  option_parser_register(opp, "-gpgpu_mem_address_mask", OPT_INT32,
                         &gpgpu_mem_address_mask,
                         "0 = old addressing mask, 1 = new addressing mask, 2 "
                         "= new add. mask + flipped bank sel and chip sel bits",
                         "0");
  option_parser_register(
      opp, "-gpgpu_memory_partition_indexing", OPT_UINT32,
      &memory_partition_indexing,
      "0 = no indexing, 1 = bitwise xoring, 2 = IPoly, 3 = custom indexing",
      "0");
}

new_addr_type linear_to_raw_address_translation::partition_address(
    new_addr_type addr_complete) const {

  new_addr_type addr;
  // Memory domain detection
  int indomain = -1; // Address is in global memory domain
  if ( m_n_mem_domains > 1 ) {
	  if ( addr_complete >= (GLOBAL_HEAP_START + MEM_DOMAIN1_OFFSET + MEM_DOMAIN0_OFFSET ) )
		  addr = addr_complete;
	  else if ( addr_complete >= (GLOBAL_HEAP_START + MEM_DOMAIN1_OFFSET) )
	  {
		  addr = addr_complete - MEM_DOMAIN1_OFFSET;
		  indomain = 1;
	  }
	  else if ( addr_complete >= (GLOBAL_HEAP_START + MEM_DOMAIN0_OFFSET) )
	  {
		  addr = addr_complete - MEM_DOMAIN0_OFFSET;
		  indomain = 0;
	  }
	  else 
    {
      addr = addr_complete; // Address is in global memory domain
    }
  } else addr = addr_complete;

  switch(indomain)
  {
    case -1: {
      if (!gap) {
        return addrdec_packbits(~(addrdec_mask[CHIP] | sub_partition_id_mask),
                                addr, 64, 0);
      } else {
        // see addrdec_tlx for explanation
        unsigned long long int partition_addr;
        partition_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
        partition_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
        // remove the part of address that constributes to the sub partition ID
        partition_addr =
            addrdec_packbits(~sub_partition_id_mask, partition_addr, 64, 0);
        return partition_addr;
      }
      break;
    }
    case 0: {
      if (!gap0) {
        return addrdec_packbits(~(addrdec_mask0[CHIP] | sub_partition_id_mask0),
                                addr, 64, 0);
      } else {
        // see addrdec_tlx for explanation
        unsigned long long int partition_addr;
        partition_addr = ((addr >> ADDR_CHIP_S) / m_n_channel0) << ADDR_CHIP_S;
        partition_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
        // remove the part of address that constributes to the sub partition ID
        partition_addr =
            addrdec_packbits(~sub_partition_id_mask0, partition_addr, 64, 0);
        return partition_addr;
      }
      break;
    }
    case 1:{
      if (!gap1) {
        return addrdec_packbits(~(addrdec_mask1[CHIP] | sub_partition_id_mask1),
                                addr, 64, 0);
      } else {
        // see addrdec_tlx for explanation
        unsigned long long int partition_addr;
        partition_addr = ((addr >> ADDR_CHIP_S) / m_n_channel1) << ADDR_CHIP_S;
        partition_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
        // remove the part of address that constributes to the sub partition ID
        partition_addr =
            addrdec_packbits(~sub_partition_id_mask1, partition_addr, 64, 0);
        return partition_addr;
      }
      break;
    }
  }
}

void linear_to_raw_address_translation::addrdec_tlx(new_addr_type addr_complete,
                                                    addrdec_t *tlx) const {
  unsigned long long int addr_for_chip, rest_of_addr, rest_of_addr_high_bits;

  new_addr_type addr;
  // Memory domain detection
  int indomain = -1; // Address is in global memory domain
  if ( m_n_mem_domains > 1 ) {
	  if ( addr_complete >= (GLOBAL_HEAP_START + MEM_DOMAIN1_OFFSET + MEM_DOMAIN0_OFFSET ) )
		  addr = addr_complete;
	  else if ( addr_complete >= (GLOBAL_HEAP_START + MEM_DOMAIN1_OFFSET) )
	  {
		  addr = addr_complete - MEM_DOMAIN1_OFFSET;
		  indomain = 1;
	  }
	  else if ( addr_complete >= (GLOBAL_HEAP_START + MEM_DOMAIN0_OFFSET) )
	  {
		  addr = addr_complete - MEM_DOMAIN0_OFFSET;
		  indomain = 0;
	  }
	  else 
    {
      addr = addr_complete; // Address is in global memory domain
    }
  } else addr = addr_complete;
  
  switch(indomain) {
    case -1: {
      if (!gap) {
        tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr,
                                     addrdec_mkhigh[CHIP], addrdec_mklow[CHIP]);
        tlx->bk = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK],
                                   addrdec_mklow[BK]);
        tlx->row = addrdec_packbits(addrdec_mask[ROW], addr,
                                    addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
        tlx->col = addrdec_packbits(addrdec_mask[COL], addr,
                                    addrdec_mkhigh[COL], addrdec_mklow[COL]);
        tlx->burst =
            addrdec_packbits(addrdec_mask[BURST], addr, addrdec_mkhigh[BURST],
                             addrdec_mklow[BURST]);
        rest_of_addr_high_bits =
            (addr >> (ADDR_CHIP_S + (log2channel + log2sub_partition)));

      } else {
        // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
        // - extract chip address using modulus of MSBs
        // - recreate the rest of the address by stitching the quotient of MSBs
        // and the LSBs
        addr_for_chip = (addr >> ADDR_CHIP_S) % m_n_channel;
        rest_of_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
        rest_of_addr_high_bits = ((addr >> ADDR_CHIP_S) / m_n_channel);
        rest_of_addr |= addr & ((1 << ADDR_CHIP_S) - 1);

        tlx->chip = addr_for_chip;
        tlx->bk = addrdec_packbits(addrdec_mask[BK], rest_of_addr,
                                   addrdec_mkhigh[BK], addrdec_mklow[BK]);
        tlx->row = addrdec_packbits(addrdec_mask[ROW], rest_of_addr,
                                    addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
        tlx->col = addrdec_packbits(addrdec_mask[COL], rest_of_addr,
                                    addrdec_mkhigh[COL], addrdec_mklow[COL]);
        tlx->burst =
            addrdec_packbits(addrdec_mask[BURST], rest_of_addr,
                             addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
      }
      break;
    }
    case 0: {
      if (!gap0) {
        tlx->chip = addrdec_packbits(addrdec_mask0[CHIP], addr,
                                     addrdec_mkhigh0[CHIP], addrdec_mklow0[CHIP]);
        tlx->bk = addrdec_packbits(addrdec_mask0[BK], addr, addrdec_mkhigh0[BK],
                                   addrdec_mklow0[BK]);
        tlx->row = addrdec_packbits(addrdec_mask0[ROW], addr,
                                    addrdec_mkhigh0[ROW], addrdec_mklow0[ROW]);
        tlx->col = addrdec_packbits(addrdec_mask0[COL], addr,
                                    addrdec_mkhigh0[COL], addrdec_mklow0[COL]);
        tlx->burst =
            addrdec_packbits(addrdec_mask0[BURST], addr, addrdec_mkhigh0[BURST],
                             addrdec_mklow0[BURST]);
        rest_of_addr_high_bits =
            (addr >> (ADDR_CHIP_S + (log2channel0 + log2sub_partition0)));

      } else {
        // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
        // - extract chip address using modulus of MSBs
        // - recreate the rest of the address by stitching the quotient of MSBs
        // and the LSBs
        addr_for_chip = (addr >> ADDR_CHIP_S) % m_n_channel0;
        rest_of_addr = ((addr >> ADDR_CHIP_S) / m_n_channel0) << ADDR_CHIP_S;
        rest_of_addr_high_bits = ((addr >> ADDR_CHIP_S) / m_n_channel0);
        rest_of_addr |= addr & ((1 << ADDR_CHIP_S) - 1);

        tlx->chip = addr_for_chip;
        tlx->bk = addrdec_packbits(addrdec_mask0[BK], rest_of_addr,
                                   addrdec_mkhigh0[BK], addrdec_mklow0[BK]);
        tlx->row = addrdec_packbits(addrdec_mask0[ROW], rest_of_addr,
                                    addrdec_mkhigh0[ROW], addrdec_mklow0[ROW]);
        tlx->col = addrdec_packbits(addrdec_mask0[COL], rest_of_addr,
                                    addrdec_mkhigh0[COL], addrdec_mklow0[COL]);
        tlx->burst =
            addrdec_packbits(addrdec_mask0[BURST], rest_of_addr,
                             addrdec_mkhigh0[BURST], addrdec_mklow0[BURST]);
      }
      break;
    }
    case 1: {
      if (!gap1) {
        tlx->chip = addrdec_packbits(addrdec_mask1[CHIP], addr,
                                     addrdec_mkhigh1[CHIP], addrdec_mklow1[CHIP]);
        tlx->bk = addrdec_packbits(addrdec_mask1[BK], addr, addrdec_mkhigh1[BK],
                                   addrdec_mklow1[BK]);
        tlx->row = addrdec_packbits(addrdec_mask1[ROW], addr,
                                    addrdec_mkhigh1[ROW], addrdec_mklow1[ROW]);
        tlx->col = addrdec_packbits(addrdec_mask1[COL], addr,
                                    addrdec_mkhigh1[COL], addrdec_mklow1[COL]);
        tlx->burst =
            addrdec_packbits(addrdec_mask1[BURST], addr, addrdec_mkhigh1[BURST],
                             addrdec_mklow1[BURST]);
        rest_of_addr_high_bits =
            (addr >> (ADDR_CHIP_S + (log2channel1 + log2sub_partition1)));

      } else {
        // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
        // - extract chip address using modulus of MSBs
        // - recreate the rest of the address by stitching the quotient of MSBs
        // and the LSBs
        addr_for_chip = (addr >> ADDR_CHIP_S) % m_n_channel1;
        rest_of_addr = ((addr >> ADDR_CHIP_S) / m_n_channel1) << ADDR_CHIP_S;
        rest_of_addr_high_bits = ((addr >> ADDR_CHIP_S) / m_n_channel1);
        rest_of_addr |= addr & ((1 << ADDR_CHIP_S) - 1);

        tlx->chip = addr_for_chip;
        tlx->bk = addrdec_packbits(addrdec_mask1[BK], rest_of_addr,
                                   addrdec_mkhigh1[BK], addrdec_mklow1[BK]);
        tlx->row = addrdec_packbits(addrdec_mask1[ROW], rest_of_addr,
                                    addrdec_mkhigh1[ROW], addrdec_mklow1[ROW]);
        tlx->col = addrdec_packbits(addrdec_mask1[COL], rest_of_addr,
                                    addrdec_mkhigh1[COL], addrdec_mklow1[COL]);
        tlx->burst =
            addrdec_packbits(addrdec_mask1[BURST], rest_of_addr,
                             addrdec_mkhigh1[BURST], addrdec_mklow1[BURST]);
      }
      break;
    }    
  }

  switch (memory_partition_indexing) {
    case CONSECUTIVE:
      // Do nothing
      break;
    case BITWISE_PERMUTATION: {
      assert(!gap);
      tlx->chip =
          bitwise_hash_function(rest_of_addr_high_bits, tlx->chip, m_n_channel);
      assert(tlx->chip < m_n_channel);
      break;
    }
    case IPOLY: {
      // assert(!gap);
      unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
      unsigned sub_partition = tlx->chip * m_n_sub_partition_in_channel +
                               (tlx->bk & sub_partition_addr_mask);
      sub_partition = ipoly_hash_function(
          rest_of_addr_high_bits, sub_partition,
          nextPowerOf2_m_n_channel * m_n_sub_partition_in_channel);

      if (gap)  // if it is not 2^n partitions, then take modular
        sub_partition =
            sub_partition % (m_n_channel * m_n_sub_partition_in_channel);

      tlx->chip = sub_partition / m_n_sub_partition_in_channel;
      tlx->sub_partition = sub_partition;
      assert(tlx->chip < m_n_channel);
      assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
      return;
      break;
    }
    case RANDOM: {
      // This is an unrealistic hashing using software hashtable
      // we generate a random set for each memory address and save the value in
      new_addr_type chip_address = (addr >> (ADDR_CHIP_S - log2sub_partition));
      tr1_hash_map<new_addr_type, unsigned>::const_iterator got =
          address_random_interleaving.find(chip_address);
      if (got == address_random_interleaving.end()) {
        unsigned new_chip_id =
            rand() % (m_n_channel * m_n_sub_partition_in_channel);
        address_random_interleaving[chip_address] = new_chip_id;
        tlx->chip = new_chip_id / m_n_sub_partition_in_channel;
        tlx->sub_partition = new_chip_id;
      } else {
        unsigned new_chip_id = got->second;
        tlx->chip = new_chip_id / m_n_sub_partition_in_channel;
        tlx->sub_partition = new_chip_id;
      }

      assert(tlx->chip < m_n_channel);
      assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
      return;
      break;
    }
    case CUSTOM:
      /* No custom set function implemented */
      // Do you custom index here
      break;
    default:
      assert("\nUndefined set index function.\n" && 0);
      break;
  }

  if ( m_n_mem_domains > 1 )
  {
	   if ( indomain == 1 )
      tlx->chip += m_n_channel0;
  }

  // combine the chip address and the lower bits of DRAM bank address to form
  // the subpartition ID
  unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
  tlx->sub_partition = tlx->chip * m_n_sub_partition_in_channel +
                       (tlx->bk & sub_partition_addr_mask);
                       
  fprintf(stderr,"TLX: %llx (%llx)\t%d\t%d\t%d\t%d\t%d\t%d\n", addr_complete, addr, indomain, tlx->chip, tlx->bk, tlx->row, tlx->col, tlx->sub_partition);
}

void linear_to_raw_address_translation::addrdec_parseoption(
    const char *option) {
  unsigned int dramid_start = 0;
  int dramid_parsed = sscanf(option, "dramid@%d", &dramid_start);
  if (dramid_parsed == 1) {
    ADDR_CHIP_S = dramid_start;
  } else {
    ADDR_CHIP_S = -1;
  }

  const char *cmapping = strchr(option, ';');
  if (cmapping == NULL) {
    cmapping = option;
  } else {
    cmapping += 1;
  }

  addrdec_mask[CHIP] = 0x0;
  addrdec_mask[BK] = 0x0;
  addrdec_mask[ROW] = 0x0;
  addrdec_mask[COL] = 0x0;
  addrdec_mask[BURST] = 0x0;

  int ofs = 63;
  while ((*cmapping) != '\0') {
    switch (*cmapping) {
      case 'D':
      case 'd':
        assert(dramid_parsed != 1);
        addrdec_mask[CHIP] |= (1ULL << ofs);
        ofs--;
        break;
      case 'B':
      case 'b':
        addrdec_mask[BK] |= (1ULL << ofs);
        ofs--;
        break;
      case 'R':
      case 'r':
        addrdec_mask[ROW] |= (1ULL << ofs);
        ofs--;
        break;
      case 'C':
      case 'c':
        addrdec_mask[COL] |= (1ULL << ofs);
        ofs--;
        break;
      case 'S':
      case 's':
        addrdec_mask[BURST] |= (1ULL << ofs);
        addrdec_mask[COL] |= (1ULL << ofs);
        ofs--;
        break;
      // ignore bit
      case '0':
        ofs--;
        break;
      // ignore character
      case '|':
      case ' ':
      case '.':
        break;
      default:
        fprintf(
            stderr,
            "ERROR: Invalid address mapping character '%c' in option '%s'\n",
            *cmapping, option);
    }
    cmapping += 1;
  }

  if (ofs != -1) {
    fprintf(stderr,
            "ERROR: Invalid address mapping length (%d) in option '%s'\n",
            63 - ofs, option);
    assert(ofs == -1);
  }
}

// Memory domains
void linear_to_raw_address_translation::init(
    unsigned int n_channel, unsigned int n_sub_partition_in_channel, unsigned int n_mem_domains) {
  unsigned i;
  unsigned long long int mask;
  unsigned int nchipbits = ::LOGB2_32(n_channel);
  log2channel = nchipbits;
  log2sub_partition = ::LOGB2_32(n_sub_partition_in_channel);
  m_n_channel = n_channel;
  m_n_sub_partition_in_channel = n_sub_partition_in_channel;
  nextPowerOf2_m_n_channel = ::next_powerOf2(n_channel);
  m_n_sub_partition_total = n_channel * n_sub_partition_in_channel;

  gap = (n_channel - ::powli(2, nchipbits));
  if (gap) {
    nchipbits++;
  }

  m_n_mem_domains = n_mem_domains;
  // Memory domain 0
  unsigned int n_channel0 = n_channel/2;
  unsigned int nchipbits0 = ::LOGB2_32(n_channel0);
  log2channel0 = nchipbits0;
  log2sub_partition0 = log2sub_partition; // No need to consider a different number of subpartitions
  m_n_channel0 = n_channel0;
  m_n_sub_partition_in_channel0 = n_sub_partition_in_channel;
  nextPowerOf2_m_n_channel0 = ::next_powerOf2(n_channel0);
  m_n_sub_partition_total0 = n_channel0 * n_sub_partition_in_channel;

  gap0 = (n_channel0 - ::powli(2, nchipbits0));
  if (gap0) {
    nchipbits0++;
  }

  // Memory domain 1
  unsigned int n_channel1 = n_channel - n_channel0;
  unsigned int nchipbits1 = ::LOGB2_32(n_channel1);
  log2channel1 = nchipbits1;
  log2sub_partition1 = log2sub_partition; // No need to consider a different number of subpartitions
  m_n_channel1 = n_channel0;
  m_n_sub_partition_in_channel1 = n_sub_partition_in_channel;
  nextPowerOf2_m_n_channel1 = ::next_powerOf2(n_channel1);
  m_n_sub_partition_total1 = n_channel1 * n_sub_partition_in_channel;

  gap1 = (n_channel1 - ::powli(2, nchipbits1));
  if (gap1) {
    nchipbits1++;
  }

printf("Memory channels %d, sub %d, chipbits %d, gap %d\n", n_channel, n_sub_partition_in_channel, nchipbits, gap);
printf("Domain 0 Memory channels %d, chipbits %d, gap %d\n", n_channel0, nchipbits0, gap0);
printf("Domain 1 Memory channels %d, chipbits %d, gap %d\n", n_channel1, nchipbits1, gap1);

  switch (gpgpu_mem_address_mask) {
    case 0:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 10;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000300;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001CFF;
      break;
    case 1:
      ADDR_CHIP_S = 13;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 2:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 3:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x000000000FFFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;

    case 14:
      ADDR_CHIP_S = 14;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 15:
      ADDR_CHIP_S = 15;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 16:
      ADDR_CHIP_S = 16;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 6:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 5:
      ADDR_CHIP_S = 5;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 100:
      ADDR_CHIP_S = 1;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;
      break;
    case 103:
      ADDR_CHIP_S = 3;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;
      break;
    case 106:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 160:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000300;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001CFF;

    default:
      break;
  }

  if (addrdec_option != NULL) addrdec_parseoption(addrdec_option);

  printf("Address mask %d, %d\n", gpgpu_mem_address_mask, ADDR_CHIP_S);
  // Memory domain 0
  addrdec_mask0[CHIP] = addrdec_mask[CHIP];
  addrdec_mask0[BK] = addrdec_mask[BK];
  addrdec_mask0[ROW] = addrdec_mask[ROW];
  addrdec_mask0[COL] = addrdec_mask[COL];
  addrdec_mask0[BURST] = addrdec_mask[BURST];

  // Memory domain 1
  addrdec_mask1[CHIP] = addrdec_mask[CHIP];
  addrdec_mask1[BK] = addrdec_mask[BK];
  addrdec_mask1[ROW] = addrdec_mask[ROW];
  addrdec_mask1[COL] = addrdec_mask[COL];
  addrdec_mask1[BURST] = addrdec_mask[BURST];

  if (ADDR_CHIP_S != -1) {
    if (!gap) {
      // number of chip is power of two:
      // - insert CHIP mask starting at the bit position ADDR_CHIP_S
      mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
      addrdec_mask[BK] =
          ((addrdec_mask[BK] & ~mask) << nchipbits) | (addrdec_mask[BK] & mask);
      addrdec_mask[ROW] = ((addrdec_mask[ROW] & ~mask) << nchipbits) |
                          (addrdec_mask[ROW] & mask);
      addrdec_mask[COL] = ((addrdec_mask[COL] & ~mask) << nchipbits) |
                          (addrdec_mask[COL] & mask);

      for (i = ADDR_CHIP_S; i < (ADDR_CHIP_S + nchipbits); i++) {
        mask = (unsigned long long int)1 << i;
        addrdec_mask[CHIP] |= mask;
      }
    }  // otherwise, no need to change the masks

    if (!gap0) {
      // number of chip is power of two:
      // - insert CHIP mask starting at the bit position ADDR_CHIP_S
      mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
      addrdec_mask0[BK] =
          ((addrdec_mask0[BK] & ~mask) << nchipbits0) | (addrdec_mask0[BK] & mask);
      addrdec_mask0[ROW] = ((addrdec_mask0[ROW] & ~mask) << nchipbits0) |
                          (addrdec_mask0[ROW] & mask);
      addrdec_mask0[COL] = ((addrdec_mask0[COL] & ~mask) << nchipbits0) |
                          (addrdec_mask0[COL] & mask);

      for (i = ADDR_CHIP_S; i < (ADDR_CHIP_S + nchipbits0); i++) {
        mask = (unsigned long long int)1 << i;
        addrdec_mask0[CHIP] |= mask;
      }
    }  // otherwise, no need to change the masks

    if (!gap1) {
      // number of chip is power of two:
      // - insert CHIP mask starting at the bit position ADDR_CHIP_S
      mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
      addrdec_mask1[BK] =
          ((addrdec_mask1[BK] & ~mask) << nchipbits1) | (addrdec_mask1[BK] & mask);
      addrdec_mask1[ROW] = ((addrdec_mask1[ROW] & ~mask) << nchipbits1) |
                          (addrdec_mask1[ROW] & mask);
      addrdec_mask1[COL] = ((addrdec_mask1[COL] & ~mask) << nchipbits1) |
                          (addrdec_mask1[COL] & mask);

      for (i = ADDR_CHIP_S; i < (ADDR_CHIP_S + nchipbits1); i++) {
        mask = (unsigned long long int)1 << i;
        addrdec_mask1[CHIP] |= mask;
      }
    }  // otherwise, no need to change the masks
  } else {
    // make sure n_channel is power of two when explicit dram id mask is used
    assert((n_channel & (n_channel - 1)) == 0);
    assert((n_channel0 & (n_channel0 - 1)) == 0);
    assert((n_channel1 & (n_channel1 - 1)) == 0);
  }
  // make sure m_n_sub_partition_in_channel is power of two
  assert((m_n_sub_partition_in_channel & (m_n_sub_partition_in_channel - 1)) ==
         0);

  addrdec_getmasklimit(addrdec_mask[CHIP], &addrdec_mkhigh[CHIP],
                       &addrdec_mklow[CHIP]);
  addrdec_getmasklimit(addrdec_mask[BK], &addrdec_mkhigh[BK],
                       &addrdec_mklow[BK]);
  addrdec_getmasklimit(addrdec_mask[ROW], &addrdec_mkhigh[ROW],
                       &addrdec_mklow[ROW]);
  addrdec_getmasklimit(addrdec_mask[COL], &addrdec_mkhigh[COL],
                       &addrdec_mklow[COL]);
  addrdec_getmasklimit(addrdec_mask[BURST], &addrdec_mkhigh[BURST],
                       &addrdec_mklow[BURST]);

  addrdec_getmasklimit(addrdec_mask0[CHIP], &addrdec_mkhigh0[CHIP],
                       &addrdec_mklow0[CHIP]);
  addrdec_getmasklimit(addrdec_mask0[BK], &addrdec_mkhigh0[BK],
                       &addrdec_mklow0[BK]);
  addrdec_getmasklimit(addrdec_mask0[ROW], &addrdec_mkhigh0[ROW],
                       &addrdec_mklow0[ROW]);
  addrdec_getmasklimit(addrdec_mask0[COL], &addrdec_mkhigh0[COL],
                       &addrdec_mklow0[COL]);
  addrdec_getmasklimit(addrdec_mask0[BURST], &addrdec_mkhigh0[BURST],
                       &addrdec_mklow0[BURST]);

  addrdec_getmasklimit(addrdec_mask1[CHIP], &addrdec_mkhigh1[CHIP],
                       &addrdec_mklow1[CHIP]);
  addrdec_getmasklimit(addrdec_mask1[BK], &addrdec_mkhigh1[BK],
                       &addrdec_mklow1[BK]);
  addrdec_getmasklimit(addrdec_mask1[ROW], &addrdec_mkhigh1[ROW],
                       &addrdec_mklow1[ROW]);
  addrdec_getmasklimit(addrdec_mask1[COL], &addrdec_mkhigh1[COL],
                       &addrdec_mklow1[COL]);
  addrdec_getmasklimit(addrdec_mask1[BURST], &addrdec_mkhigh1[BURST],
                       &addrdec_mklow1[BURST]);

  printf("addr_dec_mask[CHIP]  = %016llx \thigh:%d low:%d\n",
         addrdec_mask[CHIP], addrdec_mkhigh[CHIP], addrdec_mklow[CHIP]);
  printf("addr_dec_mask[BK]    = %016llx \thigh:%d low:%d\n", addrdec_mask[BK],
         addrdec_mkhigh[BK], addrdec_mklow[BK]);
  printf("addr_dec_mask[ROW]   = %016llx \thigh:%d low:%d\n", addrdec_mask[ROW],
         addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
  printf("addr_dec_mask[COL]   = %016llx \thigh:%d low:%d\n", addrdec_mask[COL],
         addrdec_mkhigh[COL], addrdec_mklow[COL]);
  printf("addr_dec_mask[BURST] = %016llx \thigh:%d low:%d\n",
         addrdec_mask[BURST], addrdec_mkhigh[BURST], addrdec_mklow[BURST]);

  printf("Memory domain 0\n");
  printf("addr_dec_mask[CHIP]  = %016llx \thigh:%d low:%d\n",
         addrdec_mask0[CHIP], addrdec_mkhigh0[CHIP], addrdec_mklow0[CHIP]);
  printf("addr_dec_mask[BK]    = %016llx \thigh:%d low:%d\n", addrdec_mask0[BK],
         addrdec_mkhigh0[BK], addrdec_mklow0[BK]);
  printf("addr_dec_mask[ROW]   = %016llx \thigh:%d low:%d\n", addrdec_mask0[ROW],
         addrdec_mkhigh0[ROW], addrdec_mklow0[ROW]);
  printf("addr_dec_mask[COL]   = %016llx \thigh:%d low:%d\n", addrdec_mask0[COL],
         addrdec_mkhigh0[COL], addrdec_mklow0[COL]);
  printf("addr_dec_mask[BURST] = %016llx \thigh:%d low:%d\n",
         addrdec_mask0[BURST], addrdec_mkhigh0[BURST], addrdec_mklow0[BURST]);

  printf("Memory domain 1\n");
  printf("addr_dec_mask[CHIP]  = %016llx \thigh:%d low:%d\n",
         addrdec_mask1[CHIP], addrdec_mkhigh1[CHIP], addrdec_mklow1[CHIP]);
  printf("addr_dec_mask[BK]    = %016llx \thigh:%d low:%d\n", addrdec_mask1[BK],
         addrdec_mkhigh1[BK], addrdec_mklow1[BK]);
  printf("addr_dec_mask[ROW]   = %016llx \thigh:%d low:%d\n", addrdec_mask1[ROW],
         addrdec_mkhigh1[ROW], addrdec_mklow1[ROW]);
  printf("addr_dec_mask[COL]   = %016llx \thigh:%d low:%d\n", addrdec_mask1[COL],
         addrdec_mkhigh1[COL], addrdec_mklow1[COL]);
  printf("addr_dec_mask[BURST] = %016llx \thigh:%d low:%d\n",
         addrdec_mask1[BURST], addrdec_mkhigh1[BURST], addrdec_mklow1[BURST]);

  // create the sub partition ID mask (for removing the sub partition ID from
  // the partition address)
  sub_partition_id_mask = 0;
  if (m_n_sub_partition_in_channel > 1) {
    unsigned n_sub_partition_log2 = LOGB2_32(m_n_sub_partition_in_channel);
    unsigned pos = 0;
    for (unsigned i = addrdec_mklow[BK]; i < addrdec_mkhigh[BK]; i++) {
      if ((addrdec_mask[BK] & ((unsigned long long int)1 << i)) != 0) {
        sub_partition_id_mask |= ((unsigned long long int)1 << i);
        pos++;
        if (pos >= n_sub_partition_log2) break;
      }
    }
  }
  printf("sub_partition_id_mask = %016llx\n", sub_partition_id_mask);

  sub_partition_id_mask0 = 0;
  if (m_n_sub_partition_in_channel0 > 1) {
    unsigned n_sub_partition_log20 = LOGB2_32(m_n_sub_partition_in_channel0);
    unsigned pos = 0;
    for (unsigned i = addrdec_mklow0[BK]; i < addrdec_mkhigh0[BK]; i++) {
      if ((addrdec_mask0[BK] & ((unsigned long long int)1 << i)) != 0) {
        sub_partition_id_mask0 |= ((unsigned long long int)1 << i);
        pos++;
        if (pos >= n_sub_partition_log20) break;
      }
    }
  }
  printf("sub_partition_id_mask0 = %016llx\n", sub_partition_id_mask0);

  sub_partition_id_mask1 = 0;
  if (m_n_sub_partition_in_channel1 > 1) {
    unsigned n_sub_partition_log21 = LOGB2_32(m_n_sub_partition_in_channel1);
    unsigned pos = 0;
    for (unsigned i = addrdec_mklow1[BK]; i < addrdec_mkhigh1[BK]; i++) {
      if ((addrdec_mask1[BK] & ((unsigned long long int)1 << i)) != 0) {
        sub_partition_id_mask1 |= ((unsigned long long int)1 << i);
        pos++;
        if (pos >= n_sub_partition_log21) break;
      }
    }
  }
  printf("sub_partition_id_mask1 = %016llx\n", sub_partition_id_mask1);


  if (run_test) {
    sweep_test();
  }

  if (memory_partition_indexing == RANDOM) srand(1);
}

#include "../tr1_hash_map.h"

bool operator==(const addrdec_t &x, const addrdec_t &y) {
  return (memcmp(&x, &y, sizeof(addrdec_t)) == 0);
}

bool operator<(const addrdec_t &x, const addrdec_t &y) {
  if (x.chip >= y.chip)
    return false;
  else if (x.bk >= y.bk)
    return false;
  else if (x.row >= y.row)
    return false;
  else if (x.col >= y.col)
    return false;
  else if (x.burst >= y.burst)
    return false;
  else
    return true;
}

class hash_addrdec_t {
 public:
  size_t operator()(const addrdec_t &x) const {
    return (x.chip ^ x.bk ^ x.row ^ x.col ^ x.burst);
  }
};

// a simple sweep test to ensure that two linear addresses are not mapped to the
// same raw address
void linear_to_raw_address_translation::sweep_test() const {
  new_addr_type sweep_range = 16 * 1024 * 1024;

#if tr1_hash_map_ismap == 1
  typedef tr1_hash_map<addrdec_t, new_addr_type> history_map_t;
#else
  typedef tr1_hash_map<addrdec_t, new_addr_type, hash_addrdec_t> history_map_t;
#endif
  history_map_t history_map;

  for (new_addr_type raw_addr = 4; raw_addr < sweep_range; raw_addr += 4) {
    addrdec_t tlx;
    addrdec_tlx(raw_addr, &tlx);

    history_map_t::iterator h = history_map.find(tlx);

    if (h != history_map.end()) {
      printf(
          "[AddrDec] ** Error: address decoding mapping aliases two addresses "
          "to same partition with same intra-partition address: %llx %llx\n",
          h->second, raw_addr);
      abort();
    } else {
      assert((int)tlx.chip < m_n_channel);
      // ensure that partition_address() returns the concatenated address
      if ((ADDR_CHIP_S != -1 and raw_addr >= (1ULL << ADDR_CHIP_S)) or
          (ADDR_CHIP_S == -1 and raw_addr >= (1ULL << addrdec_mklow[CHIP]))) {
        assert(raw_addr != partition_address(raw_addr));
      }
      history_map[tlx] = raw_addr;
    }

    if ((raw_addr & 0xffff) == 0) printf("%llu scaned\n", raw_addr);
  }
}

void addrdec_t::print(FILE *fp) const {
  fprintf(fp, "\tchip:%x ", chip);
  fprintf(fp, "\trow:%x ", row);
  fprintf(fp, "\tcol:%x ", col);
  fprintf(fp, "\tbk:%x ", bk);
  fprintf(fp, "\tburst:%x ", burst);
  fprintf(fp, "\tsub_partition:%x ", sub_partition);
}

static long int powli(long int x, long int y)  // compute x to the y
{
  long int r = 1;
  int i;
  for (i = 0; i < y; ++i) {
    r *= x;
  }
  return r;
}

static unsigned int LOGB2_32(unsigned int v) {
  unsigned int shift;
  unsigned int r;

  r = 0;

  shift = ((v & 0xFFFF0000) != 0) << 4;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xFF00) != 0) << 3;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xF0) != 0) << 2;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xC) != 0) << 1;
  v >>= shift;
  r |= shift;
  shift = ((v & 0x2) != 0) << 0;
  v >>= shift;
  r |= shift;

  return r;
}

// compute power of two greater than or equal to n
// https://www.techiedelight.com/round-next-highest-power-2/
unsigned next_powerOf2(unsigned n) {
  // decrement n (to handle the case when n itself
  // is a power of 2)
  n = n - 1;

  // do till only one bit is left
  while (n & n - 1) n = n & (n - 1);  // unset rightmost bit

  // n is now a power of two (less than n)

  // return next power of 2
  return n << 1;
}

static new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                      unsigned char high, unsigned char low) {
  unsigned pos = 0;
  new_addr_type result = 0;
  for (unsigned i = low; i < high; i++) {
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      result |= ((val & ((unsigned long long int)1 << i)) >> i) << pos;
      pos++;
    }
  }
  return result;
}

static void addrdec_getmasklimit(new_addr_type mask, unsigned char *high,
                                 unsigned char *low) {
  *high = 64;
  *low = 0;
  int i;
  int low_found = 0;

  for (i = 0; i < 64; i++) {
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      if (low_found) {
        *high = i + 1;
      } else {
        *high = i + 1;
        *low = i;
        low_found = 1;
      }
    }
  }
}
