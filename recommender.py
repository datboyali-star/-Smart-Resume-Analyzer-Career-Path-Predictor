from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import json
import os

class Recommender:
    def __init__(self):
        # Initialize with sample job roles and courses
        # In production, these would come from a database
        self.skill_weights = {
            "Python": 1.2, "Java": 1.1, "SQL": 1.1,
            "Machine Learning": 1.3, "AWS": 1.2,
            "Docker": 1.1, "Kubernetes": 1.2
        }
        
        self.industry_clusters = {
            "web_development": ["Python", "Java", "SQL", "Git"],
            "data_science": ["Python", "R", "Machine Learning", "Statistics"],
            "cloud_devops": ["Docker", "Kubernetes", "AWS", "Linux"]
        }
        
        self.job_roles = [
            {
                "title": "Software Engineer",
                "required_skills": ["Python", "Java", "SQL", "Git", "Agile"],
                "description": "Develop and maintain software applications"
            },
            {
                "title": "Data Scientist",
                "required_skills": ["Python", "R", "Machine Learning", "SQL", "Statistics"],
                "description": "Analyze complex data sets to drive business decisions"
            },
            {
                "title": "DevOps Engineer",
                "required_skills": ["Docker", "Kubernetes", "CI/CD", "AWS", "Linux"],
                "description": "Implement and maintain deployment infrastructure"
            }
        ]

        self.courses = [
            {
                "title": "Python Programming Masterclass",
                "skills": ["Python", "Data Structures", "Algorithms"],
                "level": "Intermediate"
            },
            {
                "title": "Machine Learning Fundamentals",
                "skills": ["Python", "Machine Learning", "Statistics"],
                "level": "Advanced"
            },
            {
                "title": "Cloud Computing Essentials",
                "skills": ["AWS", "Cloud Architecture", "DevOps"],
                "level": "Beginner"
            }
        ]

        self.vectorizer = TfidfVectorizer()

    def get_recommendations(self, analyzed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized job and course recommendations"""
        skills = analyzed_data.get('skills', [])
        experience_level = analyzed_data.get('experience_level', 'intermediate')
        industry_preference = analyzed_data.get('industry_preference', None)
        
        weighted_skills = self._apply_skill_weights(skills)
        industry_match = self._get_industry_match(skills) if not industry_preference else industry_preference
        
        return {
            'job_recommendations': self._recommend_jobs(weighted_skills, industry_match),
            'course_recommendations': self._recommend_courses(weighted_skills, skills, experience_level),
            'matched_industry': industry_match
        }

    def _apply_skill_weights(self, skills: List[str]) -> str:
        """Apply weights to skills based on market demand"""
        weighted_skills = []
        for skill in skills:
            weight = self.skill_weights.get(skill, 1.0)
            weighted_skills.extend([skill] * int(weight * 10))
        return ' '.join(weighted_skills)
    
    def _get_industry_match(self, skills: List[str]) -> str:
        """Determine the best matching industry based on skills"""
        industry_scores = defaultdict(float)
        for industry, cluster_skills in self.industry_clusters.items():
            common_skills = set(skills) & set(cluster_skills)
            industry_scores[industry] = len(common_skills) / len(cluster_skills)
        return max(industry_scores.items(), key=lambda x: x[1])[0] if industry_scores else None

    def _recommend_jobs(self, weighted_skills: str, industry_match: str) -> list:
        """Recommend job roles based on weighted skills and industry match"""
        job_texts = [' '.join(job['required_skills']) for job in self.job_roles]
        job_texts.append(weighted_skills)

        # Calculate similarity with TF-IDF and industry bonus
        tfidf_matrix = self.vectorizer.fit_transform(job_texts)
        base_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        
        # Apply industry matching bonus
        job_recommendations = []
        for idx, similarity in enumerate(base_similarities):
            if similarity > 0:
                job = self.job_roles[idx]
                industry_bonus = 0.2 if any(skill in self.industry_clusters.get(industry_match, []) 
                                          for skill in job['required_skills']) else 0
                final_score = min(similarity + industry_bonus, 1.0)
                
                job_recommendations.append({
                    **job,
                    'match_score': float(final_score),
                    'industry_alignment': industry_match if industry_bonus > 0 else None
                })
        
        # Sort by match score and return top 3
        job_recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return job_recommendations[:3]

    def _recommend_courses(self, weighted_skills: str, user_skills: list, experience_level: str) -> list:
        """Recommend courses based on skills gap analysis and experience level"""
        course_texts = [' '.join(course['skills']) for course in self.courses]
        course_texts.append(weighted_skills)

        # Calculate similarity between skills and course content
        tfidf_matrix = self.vectorizer.fit_transform(course_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

        # Get course recommendations based on skill gaps and experience level
        course_recommendations = []
        experience_levels = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
        user_level = experience_levels.get(experience_level.lower(), 1)

        for idx, similarity in enumerate(cosine_similarities):
            course = self.courses[idx]
            course_level = experience_levels.get(course['level'].lower(), 1)
            level_diff = abs(course_level - user_level)
            
            # Adjust similarity based on experience level match
            adjusted_similarity = similarity * (1 - 0.2 * level_diff)
            
            missing_skills = [skill for skill in course['skills'] 
                            if skill not in user_skills]
            
            if missing_skills:  # Recommend if there are skills to learn
                # Calculate skill importance score
                skill_importance = sum(self.skill_weights.get(skill, 1.0) 
                                     for skill in missing_skills) / len(missing_skills)
                
                course_recommendations.append({
                    **course,
                    'missing_skills': missing_skills,
                    'relevance_score': float(adjusted_similarity),
                    'skill_importance': float(skill_importance),
                    'experience_match': course['level'] == experience_level
                })

        # Sort by a combination of relevance and skill importance
        course_recommendations.sort(
            key=lambda x: (x['relevance_score'] * 0.7 + x['skill_importance'] * 0.3),
            reverse=True
        )
        return course_recommendations[:3]