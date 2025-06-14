---
### **Can your model match real clinicians in rural Kenyan healthcare?**

#### **Challenge Overview**

This challenge simulates the critical, real-world medical decisions made by nurses in Kenyan rural health settings. Participants are provided with **400 authentic clinical vignettes**, each representing a scenario faced by healthcare workers with limited resources. The task is to **predict the clinician's response** to each vignette, effectively replicating the reasoning of trained professionals.
---

### **Dataset Details**

- **400 training** and **100 test samples** of clinical prompts.
- Prompts cover a wide range of domains: **maternal health, child care, critical care, etc.**
- Each prompt contains:

  - Patient presentation
  - Nurse’s experience
  - Facility type

- Responses are real, written by expert clinicians.
- Dataset is small due to the high cost of collecting high-quality, expert-validated clinical data.

---

### **Goal**

Build an AI model that:

- Accurately **predicts clinician responses**.
- Matches the **nuance and reasoning** of real professionals.
- Can perform **well in low-resource settings**.

---

### **Evaluation Metric**

- **ROUGE Score** (measures text overlap with ground truth)
- Responses are normalized (lowercase, punctuation stripped, paragraph replaced with space).

---

### **Submission Format**

A CSV file with two columns:

```
Master_Index     Clinician
ID_XXXXXX        summary a 30 yr old...
```

---

### **Model & Deployment Constraints**

Your solution **must**:

- Be **quantized** for low memory usage.
- Run inference in **< 100ms** per vignette.
- Use **< 2 GB RAM** during inference.
- Use **≤ 1 billion parameters**.
- Train within **24 hours on an NVIDIA T4 GPU**.
- Inference should work on an **NVIDIA Jetson Nano** or similar.

---

### **Prizes**

- 🥇 1st: **\$5,000**
- 🥈 2nd: **\$3,000**
- 🥉 3rd: **\$2,000**
- **5,000 Zindi points** also available.
- Winners will be acknowledged in an upcoming publication.

---

### **Judging Criteria (For Top 10 Finalists)**

You must submit a **video** explaining your solution. Judging is based on:

1. **Clarity of explanation** – 25%
2. **Insights/feature engineering** – 15%
3. **Real-world applicability** – 25%
4. **Novelty and real-world constraints** – 25%
5. **Clean, readable code** – 10%

---

### **Rules & Requirements**

- Use **open-source** tools and libraries only.
- Max **10 submissions/day**, **300 total**.
- Max **4 people per team**.
- Data **cannot be used outside** this competition.
- If ranked in top 10:

  - Submit code within **48 hours** of request.
  - Code must reproduce leaderboard score.
  - Winners must transfer IP rights of the solution to Zindi.

---

### **Code & Reproducibility**

- Code must be:

  - Deterministic (set seeds).
  - Runnable with no paid tools or credit card trials.
  - Free of custom packages.

- If code fails to run or reproduce scores, you will be disqualified from top positions.

---

### **Disqualification Policy**

- **First offence**: 6-month ban from prizes + 2000 point deduction.
- **Second offence**: Permanent ban.

---

### **Leaderboard Mechanics**

- Public leaderboard: \~20–30% of test set.
- Private leaderboard: \~70–80%, revealed at end.
- Final scores and ranks are based on **private leaderboard**.
- Ties are broken by **earliest submission time**.

---

This challenge is a **high-impact opportunity** to build real-world, deployable AI for global healthcare—especially in **resource-constrained environments**.
