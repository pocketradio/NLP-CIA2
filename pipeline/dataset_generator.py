"""
dataset_generator.py
Generate 300 synthetic documents for the sentence ordering pipeline.
5 topics x 5 templates x 12 slot variations = 300 documents.
Each document has exactly 5 sentences in canonical (correct) order.
"""

import random
import re

# ─────────────────────────────────────────────────────────────
# TECHNOLOGY
# ─────────────────────────────────────────────────────────────
TECHNOLOGY_SLOTS = {
    'company':     ['Apple', 'Google', 'Microsoft', 'Tesla', 'Amazon', 'Meta',
                    'OpenAI', 'Samsung', 'Intel', 'Nvidia', 'Adobe', 'Salesforce'],
    'product':     ['AI assistant', 'smartphone', 'cloud platform', 'neural processor',
                    'quantum chip', 'AR headset', 'autonomous vehicle', 'language model',
                    'smart device', 'data platform', 'security suite', 'developer toolkit'],
    'feature':     ['real-time processing', 'on-device AI', 'multi-modal support',
                    'encrypted storage', 'edge computing', 'natural language understanding',
                    'advanced vision', 'predictive analytics', 'federated learning',
                    'low-latency networking', 'adaptive algorithms', 'context awareness'],
    'day':         ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'ceo':         ['the CEO', 'the chief executive', 'the president', 'the founder', 'the CTO'],
    'years':       ['two', 'three', 'four', 'five', 'six'],
    'percent':     ['15', '20', '25', '30', '35'],
    'stakeholder': ['Industry analysts', 'Investors', 'Tech journalists',
                    'Enterprise clients', 'Developers'],
    'reaction':    ['applauded the innovation', 'praised the design',
                    'highlighted the performance gains', 'noted the competitive pricing',
                    'emphasized the security features'],
    'quarter':     ['first', 'second', 'third', 'fourth'],
    'amount':      ['ten million', 'fifty million', 'one hundred million', 'two hundred million'],
}

TECHNOLOGY_TEMPLATES = [
    # Pattern 1: Product Launch
    (
        "{company} unveiled its new {product} on {day}, promising to transform how users interact with technology.",
        "The {product} is powered by {feature}, which sets it apart from existing solutions on the market.",
        "{ceo} of {company} revealed that the team spent {years} years refining the technology before launch.",
        "{stakeholder} {reaction} and called it one of the most significant releases this decade.",
        "The {product} will be available globally in the {quarter} quarter, with prices starting competitively.",
    ),
    # Pattern 2: Research Breakthrough
    (
        "Researchers at {company} announced a breakthrough in {feature} that could reshape the industry.",
        "The discovery, published after {years} years of development, demonstrated a {percent}% improvement in efficiency.",
        "{ceo} described the finding as a pivotal moment in the company's history during a press briefing.",
        "{stakeholder} {reaction}, noting that the implications extend well beyond {company}'s current product line.",
        "The technology is expected to be integrated into commercial products within the next {quarter} quarters.",
    ),
    # Pattern 3: Partnership
    (
        "{company} announced a strategic partnership on {day} to accelerate development of {product} solutions.",
        "The collaboration focuses on combining {feature} with existing infrastructure to deliver new capabilities.",
        "Both organizations have been exploring joint opportunities for {years} years before formalizing the agreement.",
        "{stakeholder} {reaction}, projecting that the partnership could generate over {amount} dollars in value.",
        "The first joint product is expected to launch in the {quarter} quarter following integration testing.",
    ),
    # Pattern 4: Security/Crisis
    (
        "{company} disclosed a security vulnerability on {day} affecting its {product} platform.",
        "The flaw was found in the {feature} module and had remained undetected for an estimated {years} years.",
        "{ceo} issued a public statement confirming the issue and outlining immediate remediation steps.",
        "{stakeholder} {reaction}, urging users to apply the emergency patch as quickly as possible.",
        "Regulators have launched an inquiry and could impose fines reaching {amount} dollars if negligence is proven.",
    ),
    # Pattern 5: Market Expansion
    (
        "{company} expanded its {product} offerings on {day} to target {percent} new market segments globally.",
        "The expansion leverages {feature} to address demand in regions previously underserved by the company.",
        "{ceo} noted that the move follows {years} years of market research and localization efforts.",
        "{stakeholder} {reaction}, predicting the expansion could add significant revenue within two years.",
        "The rollout begins in the {quarter} quarter and will reach all target markets by year end.",
    ),
]

# ─────────────────────────────────────────────────────────────
# SCIENCE
# ─────────────────────────────────────────────────────────────
SCIENCE_SLOTS = {
    'researcher':  ['Dr. Chen', 'Prof. Patel', 'Dr. Nguyen', 'Prof. Ivanova',
                    'Dr. Okonkwo', 'Prof. Martinez', 'Dr. Kim', 'Prof. Hassan',
                    'Dr. Petrov', 'Prof. Zhang', 'Dr. Williams', 'Prof. Osei'],
    'institution': ['MIT', 'Stanford University', 'Oxford University', 'CERN',
                    'NASA', 'Johns Hopkins University', 'ETH Zurich', 'Caltech',
                    'Harvard Medical School', 'Max Planck Institute',
                    'Cambridge University', 'University of Tokyo'],
    'discovery':   ['a new compound', 'a genetic marker', 'a quantum phenomenon',
                    'an enzyme pathway', 'a gravitational anomaly', 'a microbial colony',
                    'a protein structure', 'a fossil specimen', 'a neural circuit',
                    'an atmospheric pattern', 'a chemical reaction', 'a cell mechanism'],
    'field':       ['oncology', 'astrophysics', 'neuroscience', 'genomics',
                    'climate science', 'quantum physics', 'virology', 'materials science',
                    'ecology', 'biochemistry', 'immunology', 'geophysics'],
    'journal':     ['Nature', 'Science', 'Cell', 'The Lancet',
                    'PNAS', 'Physical Review Letters', 'NEJM', 'JAMA',
                    'Nature Medicine', 'Science Advances', 'eLife', 'PLOS Biology'],
    'years':       ['two', 'three', 'four', 'five', 'six'],
    'percent':     ['12', '18', '23', '31', '40'],
    'implication': ['new treatment options', 'improved diagnostic tools',
                    'cleaner energy solutions', 'better climate models',
                    'novel drug targets', 'enhanced crop yields',
                    'faster computing', 'deeper space exploration',
                    'disease prevention strategies', 'biodiversity conservation',
                    'advanced materials', 'personalized medicine'],
    'organism':    ['E. coli', 'zebrafish', 'Arabidopsis', 'Drosophila',
                    'C. elegans', 'mice', 'yeast', 'human cell lines',
                    'macaques', 'ferrets', 'marmosets', 'pig models'],
    'method':      ['CRISPR gene editing', 'cryo-electron microscopy', 'machine learning analysis',
                    'mass spectrometry', 'single-cell RNA sequencing', 'X-ray crystallography',
                    'fMRI imaging', 'GWAS studies', 'deep neural networks',
                    'electron tomography', 'optogenetics', 'nanopore sequencing'],
}

SCIENCE_TEMPLATES = [
    # Pattern 1: Discovery Announcement
    (
        "{researcher} at {institution} identified {discovery} using {method}, opening new avenues in {field}.",
        "The findings, published in {journal}, represent {years} years of meticulous laboratory investigation.",
        "Initial experiments on {organism} demonstrated a {percent}% change in key biological markers.",
        "Colleagues in the field praised the study and highlighted its potential for {implication}.",
        "The team plans to expand trials and seek funding to translate the discovery into practical applications.",
    ),
    # Pattern 2: Clinical Trial
    (
        "A clinical trial led by {researcher} at {institution} has produced promising results in {field}.",
        "Participants showed a {percent}% improvement in outcomes compared to the control group after {years} years.",
        "The trial used {method} to monitor patient responses and validate the efficacy of the treatment.",
        "Results published in {journal} have encouraged regulators to fast-track approval for {discovery}.",
        "The research team is now preparing a phase three trial with a larger and more diverse patient cohort.",
    ),
    # Pattern 3: Climate Finding
    (
        "Scientists at {institution} reported new evidence of accelerating change in {field} linked to {discovery}.",
        "Using {method}, the research team analyzed data collected over {years} years from global monitoring stations.",
        "{researcher} stated that the findings suggest a {percent}% shift in baseline measurements since records began.",
        "The study, featured in {journal}, warns of serious consequences including diminished {implication}.",
        "Policymakers have been urged to act swiftly, incorporating these findings into upcoming climate legislation.",
    ),
    # Pattern 4: Space Exploration
    (
        "A team from {institution} led by {researcher} announced the detection of {discovery} in deep space.",
        "The observation was made possible by advances in {method} deployed over the past {years} years.",
        "Spectral analysis revealed anomalies consistent with {field} theories, showing a {percent}% signal deviation.",
        "The report, accepted by {journal}, calls for dedicated missions to investigate the phenomenon further.",
        "Experts believe the finding could reshape our understanding of planetary formation and {implication}.",
    ),
    # Pattern 5: Biology Breakthrough
    (
        "{researcher}'s laboratory at {institution} has mapped {discovery} in {organism} with unprecedented resolution.",
        "Employing {method}, the team resolved structures that had eluded scientists for {years} years.",
        "The breakthrough, reported in {journal}, reveals a {percent}% structural similarity to human analogs.",
        "This opens significant potential for {implication} and could redefine treatment protocols in {field}.",
        "Collaborative groups worldwide are already applying the methodology to related systems and organisms.",
    ),
]

# ─────────────────────────────────────────────────────────────
# SPORTS
# ─────────────────────────────────────────────────────────────
SPORTS_SLOTS = {
    'team':       ['the home side', 'the reigning champions', 'the underdogs', 'the visiting squad',
                   'the local favorites', 'the title holders', 'the rising contenders',
                   'the veteran squad', 'the promoted side', 'the international outfit',
                   'the regional champions', 'the tournament debutants'],
    'player':     ['the striker', 'the midfielder', 'the captain', 'the goalkeeper',
                   'the veteran forward', 'the young defender', 'the playmaker',
                   'the winger', 'the top scorer', 'the substitute',
                   'the team leader', 'the technical director'],
    'opponent':   ['their rivals', 'the defending champions', 'the top-ranked team',
                   'a formidable opponent', 'the league leaders', 'the regional favorites',
                   'a well-drilled side', 'the tournament favorites',
                   'a historically dominant team', 'the local powerhouse',
                   'an experienced international side', 'a fast-rising contender'],
    'sport':      ['football', 'basketball', 'tennis', 'cricket',
                   'rugby', 'baseball', 'swimming', 'athletics',
                   'cycling', 'volleyball', 'hockey', 'boxing'],
    'stadium':    ['the national stadium', 'the home arena', 'the iconic venue',
                   'the newly renovated ground', 'the historic coliseum',
                   'the purpose-built facility', 'the open-air stadium',
                   'the covered arena', 'the multi-purpose complex',
                   'the downtown sports center', 'the university stadium', 'the regional ground'],
    'coach':      ['the head coach', 'the team manager', 'the technical director',
                   'the interim coach', 'the veteran trainer', 'the newly appointed manager'],
    'record':     ['the all-time scoring record', 'the season points record',
                   'the consecutive wins record', 'the career appearances record',
                   'the fastest completion record', 'the highest assist tally',
                   'the longest unbeaten run', 'the most titles won',
                   'the endurance record', 'the speed record',
                   'the accuracy record', 'the comeback record'],
    'score':      ['2-1', '3-0', '4-2', '1-0', '5-3', '2-2', '6-1', '3-1'],
    'tournament': ['the national championship', 'the continental cup', 'the world series',
                   'the annual invitational', 'the regional playoffs', 'the grand slam',
                   'the premier league', 'the open championship',
                   'the federation cup', 'the international series',
                   'the club championship', 'the youth tournament'],
    'country':    ['Brazil', 'Germany', 'Japan', 'Australia', 'Spain',
                   'France', 'India', 'USA', 'Kenya', 'South Africa',
                   'Argentina', 'New Zealand'],
}

SPORTS_TEMPLATES = [
    # Pattern 1: Match Result
    (
        "{team} claimed a dramatic {score} victory over {opponent} in {sport} at {stadium}.",
        "{player} was the standout performer, contributing decisively to the winning margin.",
        "{coach} praised the squad's resilience and tactical discipline in the post-match press conference.",
        "The result moves {team} to the top of the standings ahead of the crucial final rounds.",
        "Fans celebrated long into the night, hailing the win as one of the best performances of the season.",
    ),
    # Pattern 2: Record Breaking
    (
        "{player} shattered {record} during {tournament} held at {stadium}, leaving spectators in disbelief.",
        "The achievement surpassed a benchmark that had stood for over a decade in {sport}.",
        "{coach} called the performance extraordinary and credited the athlete's years of dedicated training.",
        "Officials from {country}'s sporting federation confirmed the record and submitted documentation to world authorities.",
        "The milestone has inspired a new generation of athletes and reinvigorated interest in the sport nationally.",
    ),
    # Pattern 3: Transfer News
    (
        "{player} has completed a high-profile transfer from {team} to {opponent}, ending months of speculation.",
        "The deal, finalized ahead of the upcoming {tournament}, is one of the biggest moves in {sport} this year.",
        "{coach} of the receiving club welcomed the signing and outlined how the player fits the tactical system.",
        "Supporters in {country} followed the transfer saga closely, with reactions split between excitement and disappointment.",
        "The player is expected to make their debut in the next fixture at {stadium} following regulatory clearance.",
    ),
    # Pattern 4: Injury Comeback
    (
        "{player} made a triumphant return to {sport} at {stadium} after overcoming a serious long-term injury.",
        "The comeback was months in the making, requiring intensive rehabilitation and close medical supervision.",
        "{coach} carefully managed the player's minutes and emphasized a gradual reintegration into the squad.",
        "The crowd at {stadium} gave a standing ovation as {player} stepped onto the field for the first time.",
        "{team} secured a {score} result on the day, with the comeback story dominating post-match coverage.",
    ),
    # Pattern 5: Tournament Win
    (
        "{team} lifted the trophy at {tournament}, defeating {opponent} in a tense {score} final at {stadium}.",
        "{player} was awarded the tournament's best performer prize for consistent displays throughout the competition.",
        "{coach} dedicated the victory to the fans and acknowledged the enormous effort from the entire squad.",
        "{country}'s media celebrated the win as a landmark moment in the nation's {sport} history.",
        "The championship triumph ensures {team} a place in next year's continental competition and boosts the league's profile.",
    ),
]

# ─────────────────────────────────────────────────────────────
# TRAVEL
# ─────────────────────────────────────────────────────────────
TRAVEL_SLOTS = {
    'destination': ['Kyoto', 'Lisbon', 'Cape Town', 'Patagonia', 'Marrakech',
                    'Queenstown', 'Reykjavik', 'Cartagena', 'Tbilisi', 'Luang Prabang',
                    'Valletta', 'Chiang Mai'],
    'country':     ['Japan', 'Portugal', 'South Africa', 'Argentina', 'Morocco',
                    'New Zealand', 'Iceland', 'Colombia', 'Georgia', 'Laos',
                    'Malta', 'Thailand'],
    'landmark':    ['the ancient temple complex', 'the hilltop fortress', 'the coastal cliffs',
                    'the floating markets', 'the medieval medina', 'the volcanic landscape',
                    'the colonial old town', 'the mountain glaciers', 'the royal palace gardens',
                    'the open-air museum', 'the cathedral square', 'the riverside promenade'],
    'activity':    ['hiking through scenic trails', 'sampling local street food',
                    'exploring hidden alleyways', 'attending a traditional festival',
                    'learning local crafts', 'kayaking along the coastline',
                    'cycling through vineyards', 'joining a cooking class',
                    'photographing wildlife', 'visiting artisan markets',
                    'stargazing in remote areas', 'discovering underground caves'],
    'season':      ['spring', 'summer', 'autumn', 'winter',
                    'the dry season', 'the shoulder season'],
    'traveler':    ['solo travelers', 'couples', 'families with children',
                    'adventure seekers', 'cultural enthusiasts', 'budget backpackers',
                    'luxury vacationers', 'digital nomads', 'retirees',
                    'honeymooners', 'group tours', 'eco-tourists'],
    'hotel':       ['a boutique guesthouse', 'a historic riad', 'a cliffside resort',
                    'a treehouse lodge', 'a converted monastery', 'a floating bungalow',
                    'a design hotel', 'a mountain chalet', 'a heritage mansion',
                    'a beach villa', 'a capsule hotel', 'an eco-retreat'],
    'food':        ['grilled seafood', 'aromatic tagine', 'fresh dim sum',
                    'wood-fired pizza', 'spiced lamb stew', 'fermented delicacies',
                    'tropical fruit platters', 'artisan cheese boards',
                    'street-side noodles', 'slow-roasted meats',
                    'pastry and espresso', 'hearty stews'],
    'currency':    ['the local currency', 'the national peso', 'the regional franc',
                    'the euro', 'the local krone', 'the national lari',
                    'the pound sterling', 'the local dinar', 'the rupiah',
                    'the baht', 'the yen', 'the local dirham'],
    'transport':   ['a scenic train journey', 'a local tuk-tuk ride', 'a ferry crossing',
                    'a domestic flight', 'a rented bicycle tour', 'a river boat cruise',
                    'a cable car ascent', 'a horse-drawn carriage ride',
                    'a guided jeep safari', 'a traditional sampan ride',
                    'a coastal road trip', 'a hot-air balloon flight'],
}

TRAVEL_TEMPLATES = [
    # Pattern 1: Destination Guide
    (
        "{destination} in {country} is one of the most captivating destinations in the region, drawing {traveler} year after year.",
        "{landmark} stands as the centerpiece of any visit, offering breathtaking views and deep cultural significance.",
        "{season} is widely considered the ideal time to visit, bringing pleasant weather and vibrant local events.",
        "{traveler} will find {hotel} to be the perfect base for exploring the city's many treasures.",
        "Evenings are best spent {activity}, where the authentic local atmosphere comes fully alive.",
    ),
    # Pattern 2: Adventure Story
    (
        "A journey to {destination} begins with {transport}, immediately immersing visitors in the landscape of {country}.",
        "The highlight of the trip is {activity}, which takes {traveler} deep into {landmark}.",
        "Local guides recommend carrying {currency} in cash, as many vendors do not accept digital payments.",
        "After a demanding day of exploration, a meal of {food} at a roadside stall provides perfect restoration.",
        "Staying at {hotel} offers a peaceful retreat, with views that transform the experience into a lasting memory.",
    ),
    # Pattern 3: Cultural Experience
    (
        "{destination}'s cultural heritage is most vividly experienced through {activity} in the heart of the city.",
        "Visiting {landmark} reveals centuries of history, architecture, and artistic tradition unique to {country}.",
        "Local artisans and vendors welcome {traveler}, offering handmade souvenirs priced in {currency}.",
        "No cultural visit is complete without tasting {food}, a staple that defines the region's gastronomic identity.",
        "{season} brings special festivals that transform {destination} into a vibrant showcase of living tradition.",
    ),
    # Pattern 4: Budget Travel
    (
        "{traveler} can explore {destination} affordably by using {transport} to navigate between major attractions.",
        "{hotel} offers excellent value accommodations, costing a fraction of mainstream alternatives in {country}.",
        "Meals of {food} at local markets keep daily expenses minimal while delivering an authentic culinary experience.",
        "The must-see {landmark} can be visited free of charge during morning hours, avoiding peak-time crowds.",
        "{activity} rounds off a budget-friendly itinerary, leaving visitors with rich memories and full wallets.",
    ),
    # Pattern 5: Luxury Experience
    (
        "Discerning {traveler} choose {destination} in {country} for an unmatched blend of luxury and authenticity.",
        "Arrival via {transport} sets the tone for an indulgent stay at {hotel}, renowned for impeccable service.",
        "Private tours of {landmark} are arranged exclusively for guests, offering intimate access unavailable to the public.",
        "Culinary highlights include a chef-prepared tasting menu featuring refined interpretations of {food}.",
        "{season} amplifies the magic of {destination}, with {activity} offered as a signature bespoke experience.",
    ),
]

# ─────────────────────────────────────────────────────────────
# BUSINESS
# ─────────────────────────────────────────────────────────────
BUSINESS_SLOTS = {
    'company':   ['TechCorp', 'GlobalVentures', 'NovaTrade', 'Apex Industries', 'FrontierGroup',
                  'Meridian Holdings', 'Catalyst Partners', 'Summit Enterprises',
                  'Vanguard Capital', 'Nexus Solutions', 'Atlas Consulting', 'Pinnacle Corp'],
    'ceo':       ['the CEO', 'the chairman', 'the managing director', 'the executive director',
                  'the chief operating officer', 'the president'],
    'rival':     ['its main competitor', 'a major rival', 'the industry leader',
                  'a key challenger', 'the incumbent market leader', 'a fast-growing startup'],
    'market':    ['Southeast Asia', 'Eastern Europe', 'Sub-Saharan Africa',
                  'Latin America', 'the Middle East', 'Central Asia',
                  'North America', 'Western Europe', 'South Asia',
                  'the Nordic region', 'the Pacific Rim', 'the Gulf states'],
    'product':   ['enterprise software', 'financial services', 'logistics platform',
                  'healthcare solutions', 'consumer goods', 'industrial equipment',
                  'digital media', 'renewable energy products', 'data analytics tools',
                  'cybersecurity services', 'cloud infrastructure', 'retail technology'],
    'amount':    ['fifty million', 'one hundred million', 'two hundred million',
                  'five hundred million', 'one billion', 'two billion'],
    'percent':   ['8', '12', '15', '20', '25', '30'],
    'quarter':   ['first', 'second', 'third', 'fourth'],
    'analyst':   ['Market analysts', 'Industry observers', 'Financial commentators',
                  'Investment banks', 'Credit agencies', 'Research firms'],
    'sector':    ['technology', 'healthcare', 'finance', 'retail',
                  'energy', 'manufacturing', 'telecommunications', 'real estate'],
}

BUSINESS_TEMPLATES = [
    # Pattern 1: Earnings Report
    (
        "{company} reported {quarter}-quarter earnings that exceeded analyst forecasts, driven by strong {product} sales.",
        "Revenue climbed {percent}% year-on-year, reflecting robust demand across {market} operations.",
        "{ceo} of {company} attributed the performance to disciplined cost management and strategic investments.",
        "{analyst} revised their target prices upward, citing the sustained momentum in the {sector} segment.",
        "The company raised full-year guidance and announced a {amount}-dollar share buyback program for investors.",
    ),
    # Pattern 2: Merger
    (
        "{company} announced a {amount}-dollar merger agreement with {rival}, creating a dominant force in {sector}.",
        "The combined entity will control approximately {percent}% of the global {product} market upon completion.",
        "{ceo} described the deal as transformational and said it would accelerate growth in {market}.",
        "{analyst} called the merger strategically sound but flagged potential regulatory hurdles in key jurisdictions.",
        "The transaction is expected to close in the {quarter} quarter pending shareholder and regulatory approvals.",
    ),
    # Pattern 3: Layoffs
    (
        "{company} announced plans to reduce its global workforce by {percent}%, affecting thousands of employees.",
        "The restructuring is aimed at realigning resources toward high-growth areas including {product} and {market}.",
        "{ceo} expressed regret over the decision but emphasized the necessity of long-term strategic realignment.",
        "{analyst} viewed the move as a necessary correction and projected annual savings of {amount} dollars.",
        "Affected employees will receive severance packages and access to outplacement services in the coming weeks.",
    ),
    # Pattern 4: Market Entry
    (
        "{company} formally entered {market} with a flagship {product} offering designed for local market dynamics.",
        "The launch follows {amount} dollars of investment in regional infrastructure and talent acquisition.",
        "{ceo} called {market} a priority growth region and projected a {percent}% revenue contribution within two years.",
        "{analyst} noted that the move positions {company} favorably against {rival} in a rapidly expanding segment.",
        "The entry strategy includes localized pricing, partnerships with regional distributors, and dedicated customer support.",
    ),
    # Pattern 5: IPO
    (
        "{company} filed for an initial public offering in the {quarter} quarter, targeting a valuation of {amount} dollars.",
        "The company has achieved {percent}% compound annual growth in {product} revenues over the past three years.",
        "{ceo} stated that the IPO proceeds would fund expansion into {market} and accelerate {sector} initiatives.",
        "{analyst} assessed the offering as attractively priced given the company's growth trajectory and market position.",
        "Trading is expected to commence on the main exchange within weeks, with strong institutional demand reported.",
    ),
]


# ─────────────────────────────────────────────────────────────
# Template filling utility
# ─────────────────────────────────────────────────────────────

def fill_template(template_str, slots):
    """
    Fill a template string with randomly chosen slot values.
    Extracts all {slot_name} placeholders and fills them from the slots dict.
    """
    # Find all slot names in the template
    slot_names = re.findall(r'\{(\w+)\}', template_str)
    # Build a mapping for this fill (pick once per slot name per sentence)
    mapping = {}
    for name in slot_names:
        if name not in mapping:
            if name in slots:
                mapping[name] = random.choice(slots[name])
            else:
                mapping[name] = name  # fallback: use slot name as literal
    return template_str.format(**mapping)


def generate_dataset(seed=42):
    """
    Generate exactly 300 synthetic documents.
    5 topics x 5 templates x 12 slot variations = 300 docs.
    Each doc: {'id': int, 'topic': str, 'template': int, 'sentences': [s1..s5]}
    """
    random.seed(seed)
    docs = []
    doc_id = 0

    all_topics = [
        ('technology', TECHNOLOGY_TEMPLATES, TECHNOLOGY_SLOTS),
        ('science',    SCIENCE_TEMPLATES,    SCIENCE_SLOTS),
        ('sports',     SPORTS_TEMPLATES,     SPORTS_SLOTS),
        ('travel',     TRAVEL_TEMPLATES,     TRAVEL_SLOTS),
        ('business',   BUSINESS_TEMPLATES,   BUSINESS_SLOTS),
    ]

    for topic_name, templates, slots in all_topics:
        for template_idx, template in enumerate(templates):
            for variation in range(12):
                filled = []
                for sent_template in template:
                    sent = fill_template(sent_template, slots)
                    # Ensure sentence starts with uppercase (required for segmentation)
                    sent = sent[0].upper() + sent[1:] if sent else sent
                    filled.append(sent)
                docs.append({
                    'id':        doc_id,
                    'topic':     topic_name,
                    'template':  template_idx,
                    'sentences': filled,   # canonical order: 5 sentences
                })
                doc_id += 1

    return docs  # exactly 300 documents


if __name__ == '__main__':
    docs = generate_dataset()
    print(f"Generated {len(docs)} documents")
    print(f"Topics: {set(d['topic'] for d in docs)}")
    print(f"\nSample document (id=0):")
    for i, s in enumerate(docs[0]['sentences']):
        print(f"  [{i}] {s}")
