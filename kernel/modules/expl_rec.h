/* SPDX-License-Identifier: GPL-2.0 */
/* In-kernel API of the expl_rec module, used by the other expliccit modules. */
#ifndef _EXPL_REC_H
#define _EXPL_REC_H

#include "expliccit_rec.h"

void expl_rec_emit(struct expl_rec_hdr *rec);

#endif /* _EXPL_REC_H */
