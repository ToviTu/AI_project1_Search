def stableMatching(advisorPreferences, studentPreferences):
    advisorAssignment = {}  # advisor id to student
    studentAssignment = {}  # student id to advisor

    """IMPLEMENT METHOD HERE"""

    # Advisors propose and students accept
    open_students = set(studentPreferences.keys())

    # proceed until all students are matched
    while open_students:

        proposals = {}

        # advisor proposes to student
        for advisor in advisorPreferences:
            if advisor not in advisorAssignment and advisorPreferences[advisor]:
                student = advisorPreferences[advisor].pop(0)
                proposals[student] = proposals.get(student, []) + [advisor]

        # no proposals left
        if not proposals:
            break

        # student accepts proposal
        for student, applicants in proposals.items():
            for applicant in applicants:
                # not yet matched
                if student not in studentAssignment:
                    studentAssignment[student] = applicant
                    advisorAssignment[applicant] = student
                    open_students.remove(student)
                # already matched: prefers better advisor
                else:
                    current_advisor = studentAssignment[student]
                    if studentPreferences[student].index(
                        applicant
                    ) < studentPreferences[student].index(current_advisor):
                        studentAssignment[student] = applicant
                        advisorAssignment[applicant] = student
                        advisorAssignment.pop(current_advisor)

    return [advisorAssignment, studentAssignment]
