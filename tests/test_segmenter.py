"""Test clause segmentation with sample rental agreement text"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clause_segmenter import ClauseSegmenter

text = """WHERE IT IS AGREED AND DECLARED AS FOLLOWS: The Landlord agrees to let out and the tenant agrees to take on rent the First floor portion of the building Plat No-182, Door No 16 New/10 Old, 24th East Street, Kamaraj Nagar, Thiruvanmiyur along with electrical and sanitary fittings and other accessories fittings and structures (hereinafter called the premises) from Jan 10 2011 at the monthly rent of Rs. 14500 (Fourteen thousand and five hundred rupees) being payable on or before 5th of every month to the Landlord. The period of this agreement shall be twelve months w.e.f Jan 10, 2011. The tenant has paid Rs. 100000(0ne lack Rupees) as advance amount for the above building and the landlord shall pay this said advance without interest to the tenant at the time of vacating the premises or within 12 months of commencement of this agreement whichever is earlier. At the termination of the period of tenancy the tenant agrees to surrender to the Landlord the vacant possession of the premises without raising any objection. This rental agreement can be terminated at any time by three months notice on either side and on such termination the tenant shall surrender the vacant possession of the premises to the Landlord. If for by any reason the tenant occupies the building for a period that includes part of a month, it is agreed that the rent will be charged on a pro-rated basis for that month. The landlord shall pay all existing and future taxes, rates and assessments in respect of the lease hold including the municipal or other tax assessed by a local authority on the value of the building or annual letting value of the building and all other rates, taxes and assessments levied by any authority whatsoever. The tenant shall pay the electricity and water supply charges for the period of time he occupies the premises. The tenant agrees to leave at the end of tenancy the premises in good condition as they are now, subject to reasonable wear and tear. The tenant also agreed not to let out the building or a portion of it to anybody else. The tenant shall not commit any act of waste in the premises. 11 The tenant also agrees to make any maintenance on the building as mutually agreed upon by the tenant and the landlord and the said expenses shall be adjusted against the rent amount due to the landlord."""

segmenter = ClauseSegmenter()
result = segmenter.segment(text, 'legal')

print(f"Total clauses: {result['total_clauses']}")
print(f"Segmentation method: {result.get('segmentation_method', 'boundaries')}")
print()

for i, clause in enumerate(result['clauses'], 1):
    print(f"{i}. [{clause['clause_type']}] {clause['title']}")
    print(f"   Risk: {clause['risk_level']}")
    print(f"   Content preview: {clause['content'][:100]}...")
    print()
