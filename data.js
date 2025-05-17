const resumeData = {
    about: {
        content: `
            I’m a cybersecurity operative with 3+ years of experience fortifying BFSI and government sectors. Specializing in Vulnerability Assessment and Penetration Testing (VAPT), Red Teaming, and Secure Network Design, I deploy cutting-edge defenses to counter advanced threats. My mission: align technical security with strategic objectives, ensuring compliance and resilience. Driven by a passion for Cybersecurity and continuous innovation, I thrive in the shadows of complex cyber challenges.
        `
    },
    skills: {
        categories: [
            {
                name: "Security Operations",
                proficiency: 85,
                details: ["SIEM Monitoring (Splunk, IBM QRadar, Seceon)", "Threat Detection", "Log Analysis"],
                badge: "security-ops-badge.png"
            },
            {
                name: "Penetration Testing",
                proficiency: 90,
                details: ["Web & Network VAPT", "Red Teaming", "Exploit Analysis (Metasploit, Burp Suite, Nessus)"],
                badge: "pentest-badge.png"
            },
            {
                name: "Threat Intelligence",
                proficiency: 80,
                details: ["CTI Analysis", "TTP Analysis (MITRE ATT&CK)", "Dark Web Research (Shodan)"],
                badge: "threat-intel-badge.png"
            },
            {
                name: "Digital Forensics",
                proficiency: 75,
                details: ["Malware Analysis", "Log Forensics (Wireshark, Autopsy)", "Incident Response"],
                badge: "forensics-badge.png"
            },
            {
                name: "Compliance & Governance",
                proficiency: 80,
                details: ["PCI-DSS, ISO 27001, NIST", "Risk Assessment", "Security Audits"],
                badge: "compliance-badge.png"
            },
            {
                name: "Networking & Systems",
                proficiency: 85,
                details: ["Firewall Config (Fortinet, Palo Alto)", "Linux/Windows Security", "Automation (Ansible, BloodHound)"],
                badge: "networking-badge.png"
            },
            {
                name: "Scripting",
                proficiency: 80,
                details: ["Python", "Bash", "PowerShell"],
                badge: "scripting-badge.png"
            }
        ]
    },
    experience: {
        timeline: [
            {
                role: "Jr. Security Engineer",
                organization: "ESDS Software Solutions, Nashik",
                period: "Dec 2022 - Feb 2024",
                achievements: [
                    "Conducted 100+ VAPT engagements, securing BFSI and government networks.",
                    "Built CERT-ESDS/CSIRT, enhancing threat hunting and incident response.",
                    "Mitigated phishing and ransomware threats using Nessus, Burp Suite, and Wireshark.",
                    "Ensured compliance with ISMS, PCI-DSS, SOC 1/2, and ISO 27001 standards."
                ]
            },
            {
                role: "SOC Analyst - Trainee",
                organization: "ESDS Software Solutions, Nashik",
                period: "Jun 2022 - Dec 2022",
                achievements: [
                    "Monitored SIEM tools (Seceon, FortiSIEM, IBM QRadar) to detect anomalies.",
                    "Assisted in incident investigations, reducing breach impact.",
                    "Trained in VAPT and vulnerability mitigation, strengthening proactive defenses."
                ]
            }
        ]
    },
    education: {
        records: [
            {
                degree: "Masters of Science in Cybersecurity",
                institution: "Dublin Business School, Dublin, Ireland",
                period: "Sept 2024 - Sept 2025",
            },
            {
                degree: "B.Tech in Computer Science & Engineering",
                institution: "Sandip University, Nashik, India",
                period: "Jun 2019 - Jun 2022",
            }
        ]
    },
    certifications: {
        badges: [
            {
                name: "Certified Cybersecurity Professional",
                issuer: "ISC2",
                period: "2023"
            },
        ]
    },
    projects: {
        missions: [
                        {
                name: "DUBSEC Conference",
                description: "Delivered a talk on SSH Tunnelling at DUBSEC, showcasing secure communication tactics.",
                link: ""
            },
            {
                name: "Secure Network Design",
                description: "Architected a fortified network for a mid-sized outsourcing firm, enhancing security posture.",
                link: ""
            },
            {
                name: "Penetration Testing Ops",
                description: "Executed 100+ VAPT missions across BFSI and government sectors, neutralizing vulnerabilities.",
                link: ""
            },
            {
                name: "Incident Response Framework",
                description: "Developed ransomware mitigation plans, ensuring rapid response and recovery.",
                link: ""
            },
            {
                name: "Cyber Training Program",
                description: "Mentored teams in ethical hacking and cybersecurity awareness, boosting defense capabilities.",
                link: ""
            },
            {
                name: "AI Disaster Recovery",
                description: "Researched AI-driven disaster recovery to enhance ransomware resilience.",
                link: ""
            },
            {
                name: "AWS-Docker-CICD",
                description: "Automated AWS instance creation and Docker deployment with nginx, streamlining CI/CD.",
                link: "https://github.com/parthbhagat1337"
            }
        ]
    },
    contact: {
        channels: {
            email: "parthbhagat5997@gmail.com",
            linkedin: "https://www.linkedin.com/in/parth-bhagat-386954113/",
            github: "https://github.com/CrashNBurn1337",
            medium: "https://medium.com/@cybercrash1337",
            }
    }
};

export default resumeData;
