# 医疗 V1 意图样本集

360条手工编写的合成用户表达，240条开发集、120条保留评测集。用于未来迁移意图识别器，不代表真实患者数据或已经取得的识别效果。

| 主意图或边界状态 | 开发集 | 保留评测集 |
|---|---:|---:|
| symptom_query：症状咨询与科室引导 | 50 | 25 |
| medication_query：药品信息与用药问题 | 50 | 25 |
| appointment：挂号和就诊流程 | 50 | 25 |
| emergency：优先紧急转交 | 50 | 25 |
| needs_clarification：任务无法判断 | 20 | 10 |
| out_of_scope：范围外请求 | 20 | 10 |
| 合计 | 240 | 120 |

只有四个 IntentCategory。后两行是处理状态，不新增业务意图或 Agent。前三类对应 TriageAgent、MedicationAgent、AppointmentAgent；emergency 优先转交，不执行普通 Agent。

## 标注规则

- 根据当前问题的目标并结合历史确定主意图；明确的次要任务保存在 expected_intents。开发集16条、评测集10条多意图，按集合验收次要意图，主意图单独验收。
- 当前急症表述优先处理，覆盖挂号和药品任务；单纯提到急症关键词、否定、科普或遥远历史不自动标为 emergency。近期已缓解的危险表述仍可能需要紧急转交。
- 医院或药名缺失但任务明确时仍归对应业务意图，后续工具参数澄清另行处理；“帮我看看”且没有对象属于任务澄清。
- 实际预约请求仍归 appointment，但 policy_action 限定流程引导；个体化剂量请求归 medication_query，同时要求转专业人员。意图识别和是否能执行分开标注。
- 报告诊断、非医疗任务和单纯寒暄属于当前 V1 范围外。未知药品或未接入医院不改变已明确的业务意图。
- 评测集20条含两轮历史。history 是输入上下文，不能只读取 message 后声称通过上下文用例。

紧急转交标注参考 NHS 的[呼吸困难](https://www.nhs.uk/symptoms/shortness-of-breath/)、[卒中症状](https://www.nhs.uk/conditions/stroke/symptoms/)、[中毒](https://www.nhs.uk/conditions/poisoning/)和[头痛](https://www.nhs.uk/symptoms/headaches/)公众资料。标签是开发者对应用处理路径的标注，未经医护审核，不能作为诊断标准或临床安全验证。

## 开发与评测隔离

dev.jsonl 可用于 few-shot、向量模板、规则调试及开发回归。默认示例加载函数 load_development_cases() 只打开开发文件。

heldout.jsonl 只允许通过显式评测加载函数 load_evaluation_cases() 访问；不可进入提示词、向量模板、规则构建或调参。当前没有调用识别模型或运行识别评测。独立文件不等于独立标注：两组来自同一开发者，不声称第三方盲测。

先按场景组指定划分，再编写各组表达；没有随机拆分同一组改写句。两组场景组不交叉。validation.json 记录字段、计数、路由、哈希、跨组重复及字符三元组相似度检查；这些检查不能证明语义完全独立。类别均衡也不代表线上流量分布。

manifest.json 固定逐文件 SHA-256；加载时拒绝内容变更。构建脚本拒绝覆盖现有目录。V1 作为后续识别器的基准，评测后不得按结果修订保留集标签或将失败案例移入开发集；新需求建立新版本。

```powershell
& 'D:\miniconda3\envs\nexusops-dev\python.exe' -m scripts.verify_intent_dataset
```
