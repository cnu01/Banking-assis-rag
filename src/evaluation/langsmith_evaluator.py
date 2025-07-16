"""
LangSmith Evaluation Framework for Banking RAG System
Implements custom evaluators for banking-specific accuracy and compliance
"""

import json
import re
from typing import List, Dict, Any, Optional
from langsmith import Client

import config


class BankingAccuracyEvaluator:
    """Evaluates accuracy of banking numerical data (rates, amounts, terms)"""
    
    def __init__(self):
        self.name = "banking_accuracy"
        self.description = "Evaluates accuracy of banking numerical data in responses"
    
    def evaluate(
        self, 
        prediction: str, 
        reference: Optional[str] = None, 
        input: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate banking numerical accuracy"""
        
        # Extract numerical data from prediction
        rates = self._extract_rates(prediction)
        amounts = self._extract_amounts(prediction)
        terms = self._extract_terms(prediction)
        
        score = 1.0
        feedback = []
        
        # Validate rate formats (should be reasonable banking rates)
        for rate in rates:
            if rate < 0 or rate > 50:  # Unreasonable rate
                score -= 0.2
                feedback.append(f"Questionable rate: {rate}%")
        
        # Validate amount formats
        for amount in amounts:
            if amount <= 0:  # Invalid amount
                score -= 0.2
                feedback.append(f"Invalid amount: ${amount}")
        
        # Check for proper disclaimers in rate-related responses
        if rates and "subject to change" not in prediction.lower():
            score -= 0.1
            feedback.append("Missing rate disclaimer")
        
        return {
            "score": max(0, score),
            "feedback": "; ".join(feedback) if feedback else "Banking data appears accurate",
            "rates_found": len(rates),
            "amounts_found": len(amounts),
            "terms_found": len(terms)
        }
    
    def _extract_rates(self, text: str) -> List[float]:
        """Extract percentage rates from text"""
        pattern = r'(\d+\.?\d*)\s*%'
        return [float(match) for match in re.findall(pattern, text)]
    
    def _extract_amounts(self, text: str) -> List[float]:
        """Extract dollar amounts from text"""
        pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        amounts = re.findall(pattern, text)
        return [float(amount.replace(',', '')) for amount in amounts]
    
    def _extract_terms(self, text: str) -> List[int]:
        """Extract loan terms from text"""
        pattern = r'(\d+)\s*(?:month|year)s?'
        return [int(match) for match in re.findall(pattern, text, re.IGNORECASE)]


class TableContextEvaluator:
    """Evaluates preservation of table context and relationships"""
    
    def __init__(self):
        self.name = "table_context"
        self.description = "Evaluates preservation of table relationships and context"
    
    def evaluate(
        self, 
        prediction: str, 
        reference: Optional[str] = None, 
        input: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate table context preservation"""
        
        score = 1.0
        feedback = []
        
        # Check for table references
        table_refs = self._find_table_references(prediction)
        
        # Check for broken cross-references
        broken_refs = self._find_broken_references(prediction)
        if broken_refs:
            score -= 0.3 * len(broken_refs)
            feedback.append(f"Broken references: {', '.join(broken_refs)}")
        
        # Check for table structure preservation
        if self._has_table_structure(prediction):
            feedback.append("Table structure preserved")
        elif table_refs:
            score -= 0.2
            feedback.append("Table references found but structure unclear")
        
        # Reward specific table citations
        if table_refs:
            feedback.append(f"Table references: {', '.join(table_refs)}")
        
        return {
            "score": max(0, score),
            "feedback": "; ".join(feedback) if feedback else "No table context issues",
            "table_references": table_refs,
            "broken_references": broken_refs
        }
    
    def _find_table_references(self, text: str) -> List[str]:
        """Find table references in text"""
        pattern = r'Table ([A-Z]?\d+\.\d+)'
        return re.findall(pattern, text)
    
    def _find_broken_references(self, text: str) -> List[str]:
        """Find potentially broken table references"""
        # Look for incomplete references like "see Table" without number
        pattern = r'see Table(?!\s+[A-Z]?\d+\.\d+)'
        return re.findall(pattern, text, re.IGNORECASE)
    
    def _has_table_structure(self, text: str) -> bool:
        """Check if text preserves table structure"""
        return '|' in text or 'APR' in text or any(
            term in text for term in ['rate', 'amount', 'term', 'fee']
        )


class ComplianceEvaluator:
    """Evaluates regulatory compliance aspects of responses"""
    
    def __init__(self):
        self.name = "compliance"
        self.description = "Evaluates regulatory compliance in banking responses"
    
    def evaluate(
        self, 
        prediction: str, 
        reference: Optional[str] = None, 
        input: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate compliance aspects"""
        
        score = 1.0
        feedback = []
        
        # Check for required disclaimers
        compliance_checks = {
            'rate_disclaimer': self._check_rate_disclaimer(prediction, input),
            'credit_approval': self._check_credit_approval(prediction, input),
            'regulatory_citation': self._check_regulatory_citation(prediction, input)
        }
        
        for check, (passed, message) in compliance_checks.items():
            if not passed:
                score -= 0.25
                feedback.append(message)
            else:
                feedback.append(f"✓ {check}")
        
        return {
            "score": max(0, score),
            "feedback": "; ".join(feedback),
            "compliance_checks": {k: v[0] for k, v in compliance_checks.items()}
        }
    
    def _check_rate_disclaimer(self, prediction: str, input_text: str) -> tuple:
        """Check for proper rate disclaimers"""
        if any(term in (input_text or "").lower() for term in ['rate', 'apr', 'cost']):
            if any(phrase in prediction.lower() for phrase in 
                   ['subject to change', 'credit approval', 'may vary']):
                return True, "Rate disclaimer present"
            return False, "Missing rate disclaimer"
        return True, "No rate disclaimer needed"
    
    def _check_credit_approval(self, prediction: str, input_text: str) -> tuple:
        """Check for credit approval language"""
        if any(term in (input_text or "").lower() for term in ['loan', 'credit', 'borrow']):
            if any(phrase in prediction.lower() for phrase in 
                   ['credit approval', 'subject to approval', 'qualification']):
                return True, "Credit approval language present"
            return False, "Missing credit approval language"
        return True, "No credit approval language needed"
    
    def _check_regulatory_citation(self, prediction: str, input_text: str) -> tuple:
        """Check for regulatory citations when needed"""
        if any(term in (input_text or "").lower() for term in 
               ['compliance', 'regulation', 'fdic', 'requirement']):
            if any(term in prediction.upper() for term in 
                   ['FDIC', 'TILA', 'BSA', 'ECOA', 'FHA']):
                return True, "Regulatory citation present"
            return False, "Missing regulatory citation"
        return True, "No regulatory citation needed"


class CrossReferenceEvaluator:
    """Evaluates handling of cross-references between documents"""
    
    def __init__(self):
        self.name = "cross_reference"
        self.description = "Evaluates cross-reference resolution accuracy"
    
    def evaluate(
        self, 
        prediction: str, 
        reference: Optional[str] = None, 
        input: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate cross-reference handling"""
        
        score = 1.0
        feedback = []
        
        # Find cross-reference requests in input
        if input_text := input:
            cross_ref_requests = self._find_cross_ref_requests(input_text)
            
            if cross_ref_requests:
                # Check if cross-references were resolved
                resolved_count = 0
                for ref in cross_ref_requests:
                    if self._is_reference_resolved(ref, prediction):
                        resolved_count += 1
                    else:
                        score -= 0.5
                        feedback.append(f"Unresolved reference: {ref}")
                
                if resolved_count == len(cross_ref_requests):
                    feedback.append("All cross-references resolved")
                else:
                    feedback.append(f"Resolved {resolved_count}/{len(cross_ref_requests)} references")
        
        return {
            "score": max(0, score),
            "feedback": "; ".join(feedback) if feedback else "No cross-reference issues"
        }
    
    def _find_cross_ref_requests(self, text: str) -> List[str]:
        """Find cross-reference requests in input"""
        patterns = [
            r'see Table ([A-Z]?\d+\.\d+)',
            r'refer to Table ([A-Z]?\d+\.\d+)',
            r'Table ([A-Z]?\d+\.\d+)'
        ]
        
        refs = []
        for pattern in patterns:
            refs.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return list(set(refs))
    
    def _is_reference_resolved(self, ref: str, prediction: str) -> bool:
        """Check if a reference was resolved in the prediction"""
        # Simple check: does the prediction contain content that appears to be from the referenced table?
        return (ref in prediction or 
                any(indicator in prediction.lower() for indicator in 
                    ['rate', 'amount', 'term', 'requirement', '%', '$']))


class LangSmithEvaluator:
    """Main evaluation orchestrator for the banking RAG system"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.client = Client()
        
        # Initialize custom evaluators
        self.evaluators = {
            'banking_accuracy': BankingAccuracyEvaluator(),
            'table_context': TableContextEvaluator(),
            'compliance': ComplianceEvaluator(),
            'cross_reference': CrossReferenceEvaluator()
        }
        
        # Test datasets
        self.test_datasets = self._create_test_datasets()
    
    def _create_test_datasets(self) -> Dict[str, List[Dict]]:
        """Create comprehensive test datasets for banking evaluation"""
        
        datasets = {
            'loan_products': [
                {
                    'input': 'What are the current personal loan rates?',
                    'expected_content': ['APR', 'rate', 'Table 1.1'],
                    'category': 'loan_rates'
                },
                {
                    'input': 'Show me the amortization calculation for a $100,000 loan',
                    'expected_content': ['Table 4.1', 'payment', 'principal', 'interest'],
                    'category': 'calculations'
                },
                {
                    'input': 'What are the FHA mortgage rates?',
                    'expected_content': ['FHA', '6.875%', 'Table 2.1'],
                    'category': 'mortgage_rates'
                }
            ],
            
            'regulatory_compliance': [
                {
                    'input': 'What are the FDIC capital requirements?',
                    'expected_content': ['Table 1.1', 'Tier 1', 'capital ratio'],
                    'category': 'fdic_requirements'
                },
                {
                    'input': 'Tell me about BSA reporting requirements',
                    'expected_content': ['CTR', '$10,000', 'Table 2.1'],
                    'category': 'bsa_compliance'
                },
                {
                    'input': 'What are the TILA disclosure requirements?',
                    'expected_content': ['APR', 'Table 3.1', 'disclosure'],
                    'category': 'tila_requirements'
                }
            ],
            
            'table_cross_references': [
                {
                    'input': 'What information is in Table 1.1?',
                    'expected_content': ['personal loan', 'APR', 'terms'],
                    'category': 'direct_table_reference'
                },
                {
                    'input': 'The document mentions Table 2.2 - what does it contain?',
                    'expected_content': ['ARM', 'adjustable rate', 'initial rate'],
                    'category': 'cross_reference_resolution'
                },
                {
                    'input': 'Compare rates in Table M.1 and Table M.3',
                    'expected_content': ['conventional', 'jumbo', 'rate'],
                    'category': 'multi_table_comparison'
                }
            ]
        }
        
        return datasets
    
    def run_evaluation(self):
        """Run comprehensive evaluation across all test datasets"""
        
        print("🔍 Starting LangSmith Banking RAG Evaluation")
        print("=" * 60)
        
        all_results = {}
        
        for dataset_name, test_cases in self.test_datasets.items():
            print(f"\n📊 Evaluating: {dataset_name}")
            print("-" * 40)
            
            dataset_results = []
            
            for i, test_case in enumerate(test_cases):
                print(f"\n🧪 Test Case {i+1}: {test_case['category']}")
                print(f"❓ Question: {test_case['input']}")
                
                try:
                    # Get prediction from RAG system
                    prediction = self.rag_system.ask_question(
                        test_case['input'], 
                        use_conversation=False
                    )
                    
                    print(f"💡 Answer: {prediction[:100]}...")
                    
                    # Run evaluations
                    eval_results = {}
                    for eval_name, evaluator in self.evaluators.items():
                        result = evaluator.evaluate(
                            prediction=prediction,
                            input=test_case['input']
                        )
                        eval_results[eval_name] = result
                        print(f"📈 {eval_name}: {result['score']:.2f} - {result['feedback']}")
                    
                    # Check expected content
                    content_score = self._check_expected_content(
                        prediction, test_case['expected_content']
                    )
                    eval_results['content_coverage'] = content_score
                    
                    dataset_results.append({
                        'test_case': test_case,
                        'prediction': prediction,
                        'evaluations': eval_results
                    })
                    
                except Exception as e:
                    print(f"❌ Error in test case: {str(e)}")
                    dataset_results.append({
                        'test_case': test_case,
                        'error': str(e)
                    })
            
            all_results[dataset_name] = dataset_results
        
        # Generate summary report
        self._generate_evaluation_report(all_results)
        
        return all_results
    
    def _check_expected_content(self, prediction: str, expected_content: List[str]) -> Dict[str, Any]:
        """Check if prediction contains expected content"""
        
        found_content = []
        missing_content = []
        
        for content in expected_content:
            if content.lower() in prediction.lower():
                found_content.append(content)
            else:
                missing_content.append(content)
        
        score = len(found_content) / len(expected_content) if expected_content else 1.0
        
        return {
            'score': score,
            'found_content': found_content,
            'missing_content': missing_content,
            'feedback': f"Found {len(found_content)}/{len(expected_content)} expected items"
        }
    
    def _generate_evaluation_report(self, results: Dict[str, Any]):
        """Generate comprehensive evaluation report"""
        
        print("\n" + "=" * 60)
        print("📋 BANKING RAG EVALUATION REPORT")
        print("=" * 60)
        
        total_tests = 0
        total_scores = {'banking_accuracy': [], 'table_context': [], 
                       'compliance': [], 'cross_reference': [], 'content_coverage': []}
        
        for dataset_name, dataset_results in results.items():
            print(f"\n📊 Dataset: {dataset_name}")
            print("-" * 30)
            
            for result in dataset_results:
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                total_tests += 1
                evaluations = result['evaluations']
                
                for eval_name, eval_result in evaluations.items():
                    score = eval_result.get('score', 0)
                    total_scores[eval_name].append(score)
        
        # Calculate averages
        print(f"\n📈 OVERALL PERFORMANCE ({total_tests} tests)")
        print("-" * 40)
        
        for eval_name, scores in total_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"{eval_name.replace('_', ' ').title()}: {avg_score:.2f} ({len(scores)} tests)")
        
        # Success criteria check
        print("\n✅ SUCCESS CRITERIA CHECK")
        print("-" * 30)
        
        accuracy_avg = sum(total_scores['banking_accuracy']) / len(total_scores['banking_accuracy']) if total_scores['banking_accuracy'] else 0
        table_avg = sum(total_scores['table_context']) / len(total_scores['table_context']) if total_scores['table_context'] else 0
        compliance_avg = sum(total_scores['compliance']) / len(total_scores['compliance']) if total_scores['compliance'] else 0
        
        criteria = [
            ("Banking Accuracy ≥ 0.95", accuracy_avg >= 0.95),
            ("Table Context ≥ 0.90", table_avg >= 0.90),
            ("Compliance ≥ 0.95", compliance_avg >= 0.95),
            ("Total Tests Completed", total_tests > 0)
        ]
        
        for criterion, passed in criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{criterion}: {status}")
        
        print("\n🎯 Evaluation completed!")
        print("View detailed traces at: https://smith.langchain.com/")
        print(f"Project: {config.LANGSMITH_PROJECT}")
        
        return results 